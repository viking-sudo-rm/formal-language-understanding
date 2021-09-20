"""Implementation of rational speech acts (RSA) model for pragmatics.

A `RationalDialog` represents a conversation. A `RationalAgent` represents a participant in a conversation.

Some references:
    * https://www.problang.org/chapters/01-introduction.html
    * https://wmonroeiv.github.io/pubs/yuan2018understanding.pdf
"""

from typing import Any, Callable, Iterable, List
import torch
from torch import Tensor
from torch.distributions import Categorical


INF = 1e9


def listify(callback_fn: Callable[..., Iterable[Any]]) -> Callable[..., List[Any]]:
    """Convert a generator to a function returning a list."""

    def _wrapper(*args, **kwargs):
        return list(callback_fn(*args, **kwargs))

    return _wrapper


class RationalAgent:
    """Represents a participant in a dialog."""

    def __init__(
        self,
        belief_state: Tensor,  # Belief of this agent.
        inferred_belief_state: Tensor = None,  # Inferred belief of another agent.
        temp: float = 1.0,
        n_iter: int = 1,
    ):
        self.belief_state = belief_state
        self.inferred_belief_state = inferred_belief_state or torch.ones_like(
            belief_state
        ) / len(belief_state)
        self.temp = temp
        self.n_iter = n_iter

    def speak(self, dialog: "RationalDialog") -> Tensor:
        """Return a distribution over utterances to produce of size (n_utterances,)."""
        speak_probs = dialog.truth_values / dialog.truth_values.sum(axis=0, keepdim=True)
        for _ in range(self.n_iter):
            listen_probs = self.listen_step(speak_probs, self.inferred_belief_state)
            speak_probs = self.speak_step(listen_probs, dialog.costs)
        state_idx = Categorical(self.belief_state).sample()
        return speak_probs[:, state_idx]

    def listen(self, dialog: "RationalDialog", utter_idx: int) -> Tensor:
        """Return a distribution over inferred world states of size (n_states,)."""
        listen_probs = dialog.truth_values / dialog.truth_values.sum(axis=1, keepdim=True)
        # listen_probs = self.listen_step(dialog.truth_values, self.belief_state)
        for _ in range(self.n_iter):
            speak_probs = self.speak_step(listen_probs, dialog.costs)
            listen_probs = self.listen_step(speak_probs, self.belief_state)
        return listen_probs[utter_idx, :]

    def speak_step(self, listen_probs: Tensor, costs: Tensor) -> Tensor:
        """Represents a speaker step in the RSA process.

        Takes and returns (n_utterances, n_worlds). The input is a distribution over dim1; output over dim0.
        """
        surprisals = (listen_probs + 1e-6).log()
        surprisals = torch.where(
            surprisals < INF, surprisals, INF * torch.ones_like(surprisals)
        )
        energies = surprisals - costs.unsqueeze(dim=1)
        return (self.temp * energies).softmax(dim=0)

    def listen_step(self, speak_probs: Tensor, prior: Tensor) -> Tensor:
        """Represents a listener step in the RSA process.

        Takes and returns (n_utterances, n_worlds). The input is a distribution over dim0; output over dim1.
        """
        scores = speak_probs * prior.unsqueeze(dim=0)
        return scores / scores.sum(dim=1, keepdim=True)


class RationalDialog:
    """Speaks or listens based on the RSA model."""

    def __init__(
        self,
        utterances: List[Any],
        truth_values: Tensor,
        costs: Tensor,
        speaker: RationalAgent,
        listener: RationalAgent = None,
    ):
        self.utterances = utterances
        self.truth_values = truth_values
        self.costs = costs
        self.speaker = speaker
        self.listener = listener

    @listify
    def sample_monologue(self, length: int = 5) -> List[str]:
        """Generate a monologue from `self.speaker` attempting to convey their belief state."""
        for _ in range(length):
            # Have the speaker say something.
            utter_probs = self.speaker.speak(self)
            utter_idx = Categorical(utter_probs).sample()
            yield self.utterances[utter_idx]
            # TODO: Have the speaker update their prior of the listener's model??
            # new_prior = self.speaker.listen(self, utter_idx)
            # self.speaker.inferred_belief_state = new_prior
