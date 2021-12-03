"""Implementation of rational speech acts (RSA) model for pragmatics.

A `RationalSpeechActs` represents a conversation. A `RationalAgent` represents a participant in a conversation.

Some references:
    1. Tutorial: https://www.problang.org/chapters/01-introduction.html
    2. Foundational: http://langcog.stanford.edu/papers_new/goodman-2016-underrev.pdf
    3. Access/observations: https://onlinelibrary.wiley.com/doi/full/10.1111/tops.12007
    3. Other: https://wmonroeiv.github.io/pubs/yuan2018understanding.pdf
"""

from typing import Any, Callable, Iterable, List, NamedTuple, Optional, Tuple
import torch
from torch import Tensor
from torch.distributions import Categorical


INF = 1e9


def listify(callback_fn: Callable[..., Iterable[Any]]) -> Callable[..., List[Any]]:
    """Convert a generator to a function returning a list."""

    def _wrapper(*args, **kwargs):
        return list(callback_fn(*args, **kwargs))

    return _wrapper


def _uniform(num_items):
    return torch.ones(num_items) / num_items


class RationalSpeechActs(NamedTuple):
    """Represents data about the dialog used by the RSA model."""

    utterances: List[Any]
    truth_values: Tensor
    costs: Tensor

    @property
    def num_utterances(self):
        return self.truth_values.size(0)

    @property
    def num_worlds(self):
        return self.truth_values.size(1)


class RationalAgent:
    """Represents a participant in a dialog."""

    def __init__(
        self,
        rsa: RationalSpeechActs,
        temp: float = 1.0,
    ):
        self.rsa = rsa
        self.temp = temp

    def speak_step(self, listen_probs: Tensor) -> Tensor:
        """Represents a speaker step in the RSA recursion.

        Takes and returns (n_utterances, n_worlds). The input is a distribution over dim1; output over dim0."""
        utilities = (listen_probs + 1e-6).log()
        utilities = torch.where(utilities > -INF, utilities, -INF * torch.ones_like(utilities))
        energies = utilities - self.rsa.costs.unsqueeze(dim=1)
        return (self.temp * energies).softmax(dim=0)
    
    def modal_speak_step(self, listen_probs: Tensor, belief_state: Tensor) -> Tensor:
        """Represents a final, modalized speaker step in the RSA recursion.

        Mathematically, utility here reflects the cross-entropy between the speaker and listener belief states.
        
        Returns a distribution (n_utterances,).

        When `belief_state` is one-hot, this reduces to computing `speak_step` and selecting the true world."""
        utilities = (listen_probs + 1e-6).log()
        utilities = torch.where(utilities > -INF, utilities, -INF * torch.ones_like(utilities))
        exp_utilities = (utilities * belief_state.unsqueeze(dim=0)).sum(dim=1)
        energies = exp_utilities - self.rsa.costs
        return (self.temp * energies).softmax(dim=0)

    def listen_step(self, speak_probs: Tensor, prior: Tensor) -> Tensor:
        """Represents a listener step in the RSA process.

        Takes and returns (n_utterances, n_worlds). The input is a distribution over dim0; output over dim1."""
        scores = speak_probs * prior.unsqueeze(dim=0)
        return scores / scores.sum(dim=1, keepdim=True)

    def speak(
        self,
        inferred_belief_state: Optional[Tensor] = None,
    ) -> Tensor:
        """Return a conditional distribution over utterances to produce in every world.
        
        Returns: (n_utterances, n_worlds)"""
        inferred_belief_state = (
            _uniform(self.rsa.num_worlds)
            if inferred_belief_state is None
            else inferred_belief_state
        )
        scores = self.rsa.truth_values
        speak_probs = scores / scores.sum(dim=1, keepdim=True)
        listen_probs = self.listen_step(speak_probs, inferred_belief_state)      
        return self.speak_step(listen_probs)

    def modal_speak(
        self,
        belief_state: Tensor,
        inferred_belief_state: Optional[Tensor] = None,
    ) -> Tensor:
        """Return a conditional distribution over utterances to produce in order to maximize expected utility.
        
        Returns: (n_utterances,)"""
        belief_state = (
            _uniform(self.rsa.num_worlds) if belief_state is None else belief_state
        )
        inferred_belief_state = (
            _uniform(self.rsa.num_worlds)
            if inferred_belief_state is None
            else inferred_belief_state
        )
        scores = self.rsa.truth_values
        speak_probs = scores / scores.sum(dim=1, keepdim=True)
        listen_probs = self.listen_step(speak_probs, inferred_belief_state)      
        return self.modal_speak_step(listen_probs, belief_state)

    def listen(
        self,
        belief_state: Optional[Tensor] = None,
    ) -> Tensor:
        """Return a conditional distribution over inferred world states."""
        belief_state = (
            _uniform(self.rsa.num_worlds) if belief_state is None else belief_state
        )
        scores = self.rsa.truth_values * belief_state.unsqueeze(dim=0)
        listen_probs = scores / scores.sum(dim=0, keepdim=True)
        speak_probs = self.speak_step(listen_probs)
        return self.listen_step(speak_probs, belief_state)

    @listify
    def sample_monologue(
        self,
        speaker_belief_state: Tensor,
        listener_belief_state: Optional[Tensor] = None,  # Inferred by the speaker.
        length: int = 5,
        update_prior: bool = True,
    ) -> List[str]:
        """Generate a monologue from `self.speaker` attempting to convey their belief state."""
        listener_belief_state = (
            _uniform(self.rsa.num_worlds)
            if listener_belief_state is None
            else listener_belief_state
        )
        for _ in range(length):
            # Sample an utterance from the speaker model.
            utter_probs = self.modal_speak(speaker_belief_state, listener_belief_state)
            utter_idx = Categorical(utter_probs).sample()
            yield self.rsa.utterances[utter_idx]

            # Update the listener belief based on the utterance.
            if update_prior:
                listen_probs = self.listen(listener_belief_state)
                listener_belief_state = listen_probs[utter_idx, :]

    # TODO: What's the right way to model a back-and-forth dialog?
    # @listify
    # def sample_dialog(
    #     self,
    #     state0: Tensor,
    #     state1: Tensor,
    #     length: int = 5,
    # ) -> List[Tuple[int, str]]:
    #     """Simulate a dialog where each person tries to infer each other's state, but their beliefs don't change.

    #     Each participant starts off assuming the other shares their beliefs.
    #     """
    #     inferred_state0 = _uniform(self.dialog.num_worlds)  # state1
    #     inferred_state1 = _uniform(self.dialog.num_worlds)  # state0
    #     for _ in range(length):
    #         speak_probs = self.speak(state0, inferred_state1)
    #         breakpoint()
    #         utter_idx = _sample_utterance(speak_probs, state0)
    #         yield 0, self.dialog.utterances[utter_idx]
    #         listen_probs = self.listen(state1, inferred_state0)
    #         inferred_state0 = listen_probs[utter_idx, :]

    #         speak_probs = self.speak(state1, inferred_state0)
    #         utter_idx = _sample_utterance(speak_probs, state1)
    #         yield 1, self.dialog.utterances[utter_idx]
    #         listen_probs = self.listen(state0, inferred_state1)
    #         inferred_state1 = listen_probs[utter_idx, :]
