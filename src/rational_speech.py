"""Implementation of rational speech acts (RSA) model for pragmatics.

A `RationalDialog` represents a conversation. A `RationalAgent` represents a participant in a conversation.

Some references:
    1. https://www.problang.org/chapters/01-introduction.html
    2. http://langcog.stanford.edu/papers_new/goodman-2016-underrev.pdf
    3. https://wmonroeiv.github.io/pubs/yuan2018understanding.pdf
"""

from typing import Any, Callable, Iterable, List, NamedTuple, Optional
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
        temp: float = 1.0,
        n_iter: int = 1,
    ):
        self.dialog = dialog
        self.temp = temp
        self.n_iter = n_iter

    def speak(self, inferred_belief_state: Optional[Tensor] = None) -> Tensor:
        """Return a conditional distribution over utterances to produce."""
        inferred_belief_state = (
            _uniform(self.dialog.num_worlds)
            if inferred_belief_state is None
            else inferred_belief_state
        )
        scores = self.dialog.truth_values
        speak_probs = scores / scores.sum(dim=1, keepdim=True)
        for _ in range(self.n_iter):
            listen_probs = self.listen_step(speak_probs, inferred_belief_state)
            speak_probs = self.speak_step(listen_probs, self.dialog.costs)
        return speak_probs

    def listen(self, belief_state: Optional[Tensor] = None) -> Tensor:
        """Return a conditional distribution over inferred world states."""
        belief_state = (
            _uniform(self.dialog.num_worlds) if belief_state is None else belief_state
        )
        scores = self.dialog.truth_values * belief_state.unsqueeze(dim=0)
        listen_probs = scores / scores.sum(dim=0, keepdim=True)
        for _ in range(self.n_iter):
            speak_probs = self.speak_step(listen_probs, self.dialog.costs)
            listen_probs = self.listen_step(speak_probs, belief_state)
        return listen_probs

    def speak_step(self, listen_probs: Tensor, costs: Tensor) -> Tensor:
        """Represents a speaker step in the RSA process.

        Takes and returns (n_utterances, n_worlds). The input is a distribution over dim1; output over dim0.
        """
        surprisals = (listen_probs + 1e-6).log()
        # surprisals = torch.where(
        #     surprisals < INF, surprisals, INF * torch.ones_like(surprisals)
        # )
        energies = surprisals - costs.unsqueeze(dim=1)
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
        listener_belief_state: Optional[Tensor] = None,
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
            speak_probs = self.speak(listener_belief_state)
            world_idx = Categorical(speaker_belief_state).sample()
            utter_probs = speak_probs[:, world_idx]
            utter_idx = Categorical(utter_probs).sample()
            yield self.dialog.utterances[utter_idx]

            # Update the listener belief based on the utterance.
            if update_prior:
                listen_probs = self.listen(listener_belief_state)
                listener_belief_state = listen_probs[utter_idx, :]

            # TODO: Is this process implicitly minimizing KL divergence between `speaker_belief_state` and `listener_belief_state?`
