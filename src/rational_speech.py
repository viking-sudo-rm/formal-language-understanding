import torch
from torch.distributions import Categorical
import matplotlib.pyplot as plt


def listify(callback_fn):
    """Convert a generator to a function returning a list."""
    def _wrapper(*args, **kwargs):
        return list(callback_fn(*args, **kwargs))
    return _wrapper


class RationalSpeaker:
    """Implements the recursive step of an RSA speaker."""

    def __init__(self, temp=1.):
        self.temp = temp

    def __call__(self, listen_probs, costs):
        """Takes/returns (n_utterances, n_states)."""
        surprisals = -(listen_probs + 1e-9).log()
        energies = surprisals - costs.unsqueeze(dim=1)
        return (self.temp * energies).softmax(dim=0)


class RationalListener:
    """Implements the recursive step of an RSA listener."""

    def __init__(self, prior=None, temp=1.):
        self.temp = temp
        self.prior = prior.unsqueeze(dim=0) if prior else None
    
    def __call__(self, speak_probs):
        """Takes/returns (n_utterances, n_states)."""
        if self.prior is None:
            scores = speak_probs
        else:
            scores = speak_probs * self.prior
        return scores / scores.sum(dim=1, keepdim=True)


class RationalDialog:
    """Speaks or listens based on the RSA model."""

    def __init__(self, speaker, listener, truth_values, costs, n_iter=1):
        self.speaker = speaker
        self.listener = listener
        self.truth_values = truth_values
        self.costs = costs
        self.n_iter = n_iter

    def speak(self):
        """Get two tensors (n_utterances, n_states) representing:
                1. Inferred distribution over listener belief state.
                2. Distribution over utterances to produce.
        """
        speak_probs = self.speaker(self.truth_values, self.costs)
        for _ in range(self.n_iter):
            listen_probs = self.listener(speak_probs)
            speak_probs = self.speaker(listen_probs, self.costs)
        return listen_probs, speak_probs
    
    def listen(self):
        """Get two tensors (n_utterances, n_states) representing:
                1. Distribution over belief state.
                2. Inferred distribution over speaker utterances."""
        listen_probs = self.listener(self.truth_values)
        for _ in range(self.n_iter):
            speak_probs = self.speaker(listen_probs, self.lengths)
            listen_probs = self.listener(speak_probs)
        return listen_probs


@listify
def sample_document(utterances, truth_values, costs, belief_state, length=5, temp=1.):
    """Sample a document."""
    belief_state = belief_state.unsqueeze(dim=0)
    dialog = RationalDialog(RationalSpeaker(temp=temp), RationalListener(temp=temp), truth_values, costs)
    for _ in range(length):
        listen_probs, speak_probs = dialog.speak()
        # First sample a world from the speaker's belief state.
        state_idx = Categorical(belief_state).sample()
        # Then sample an utterance based on the distribution in this world.
        utter_probs = speak_probs[:, state_idx].flatten()
        utter_idx = Categorical(utter_probs).sample()
        new_prior = listen_probs[utter_idx, :]
        dialog.listener.prior = new_prior
        yield utterances[utter_idx]


if __name__ == "__main__":
    with torch.no_grad():
        utterances = ["You are cool.", "You are smart."]
        truth_values = torch.tensor([[1, 1], [1, 0]])
        costs = torch.tensor([len(s) for s in utterances])
        belief_state = torch.ones(2) / 2
        document = sample_document(utterances, truth_values, costs, belief_state)
        print("\n".join(document))
        # Could now update the prior using listener probs. 
