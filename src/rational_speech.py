"""Implementation of rational speech acts (RSA) model for pragmatics.

A `RationalDialog` represents a conversation. A `RationalAgent` represents a participant in a conversation.

Some references:
    1. https://www.problang.org/chapters/01-introduction.html
    2. http://langcog.stanford.edu/papers_new/goodman-2016-underrev.pdf
    3. https://wmonroeiv.github.io/pubs/yuan2018understanding.pdf
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


class RationalDialog(NamedTuple):
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
        dialog: RationalDialog,
        utility: str = "surprisal",
        temp: float = 1.0,
        n_iter: int = 1,
    ):
        self.dialog = dialog
        self.utility = utility
        self.temp = temp
        self.n_iter = n_iter

    def speak(
        self,
        belief_state: Optional[Tensor] = None,  # Only used with KL divergence utility.
        inferred_belief_state: Optional[Tensor] = None,
    ) -> Tensor:
        """Return a conditional distribution over utterances to produce."""
        belief_state = (
            _uniform(self.dialog.num_worlds) if belief_state is None else belief_state
        )
        inferred_belief_state = (
            _uniform(self.dialog.num_worlds)
            if inferred_belief_state is None
            else inferred_belief_state
        )
        scores = self.dialog.truth_values
        speak_probs = scores / scores.sum(dim=1, keepdim=True)
        for _ in range(self.n_iter):
            listen_probs = self.listen_step(speak_probs, inferred_belief_state)
            speak_probs = self.speak_step(listen_probs, belief_state)
        return speak_probs

    def listen(
        self,
        belief_state: Optional[Tensor] = None,
        inferred_belief_state: Optional[Tensor] = None,
    ) -> Tensor:
        """Return a conditional distribution over inferred world states."""
        belief_state = (
            _uniform(self.dialog.num_worlds) if belief_state is None else belief_state
        )
        inferred_belief_state = (
            _uniform(self.dialog.num_worlds)
            if inferred_belief_state is None
            else inferred_belief_state
        )
        scores = self.dialog.truth_values * belief_state.unsqueeze(dim=0)
        listen_probs = scores / scores.sum(dim=0, keepdim=True)
        for _ in range(self.n_iter):
            speak_probs = self.speak_step(listen_probs, inferred_belief_state)
            listen_probs = self.listen_step(speak_probs, belief_state)
        return listen_probs

    def speak_step(self, listen_probs: Tensor, prior: Tensor) -> Tensor:
        """Represents a speaker step in the RSA process.

        Takes and returns (n_utterances, n_worlds). The input is a distribution over dim1; output over dim0.
        """
        if self.utility == "surprisal":
            utilities = (listen_probs + 1e-3).log()
            utilities = torch.where(utilities > -INF, utilities, -INF * torch.ones_like(utilities))
        elif self.utility == "kl_div":
            # FIXME: KL divergence doesn't make sense here. Would need to reorganize this stuff a bit.
            terms = listen_probs * (prior.unsqueeze(dim=0) / listen_probs).log()
            utilities = -terms.sum(dim=1)
            breakpoint()
        else:
            return NotImplemented
        energies = utilities - self.dialog.costs.unsqueeze(dim=1)
        return (self.temp * energies).softmax(dim=0)

    def listen_step(self, speak_probs: Tensor, prior: Tensor) -> Tensor:
        """Represents a listener step in the RSA process.

        Takes and returns (n_utterances, n_worlds). The input is a distribution over dim0; output over dim1.
        """
        scores = speak_probs * prior.unsqueeze(dim=0)
        return scores / scores.sum(dim=1, keepdim=True)

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
            _uniform(self.dialog.num_worlds)
            if listener_belief_state is None
            else listener_belief_state
        )
        for _ in range(length):
            # Sample an utterance from the speaker model.
            speak_probs = self.speak(speaker_belief_state, listener_belief_state)
            utter_idx = _sample_utterance(speak_probs, speaker_belief_state)
            yield self.dialog.utterances[utter_idx]

            # Update the listener belief based on the utterance.
            if update_prior:
                listen_probs = self.listen(listener_belief_state, None)
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


def _sample_utterance(speak_probs: Tensor, belief_state: Tensor) -> int:
    world_idx = Categorical(belief_state).sample()
    utter_probs = speak_probs[:, world_idx]
    return Categorical(utter_probs).sample()
