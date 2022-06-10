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
import math


INF = 1e9


def listify(callback_fn: Callable[..., Iterable[Any]]) -> Callable[..., List[Any]]:
    """Convert a generator to a function returning a list."""

    def _wrapper(*args, **kwargs):
        return list(callback_fn(*args, **kwargs))

    return _wrapper


def _uniform(num_items):
    return torch.ones(num_items) / num_items


def _fix_nans(probs, dim):
    """If a full row is all NaN, return zeros instead."""
    zeros = torch.zeros_like(probs)
    return torch.where(probs.isnan().all(dim=dim, keepdim=True), zeros, probs)


class RationalSpeechActs(NamedTuple):
    """Represents data about the dialog used by the RSA model."""

    utterances: List[Any]
    truth_values: Tensor
    costs: Tensor
    errors: Optional[Tensor] = None
    syntax: Optional[Any] = None

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
        depth: int = 1,
        noisy: bool = False,
        conditional_independence: bool = False,
    ):
        self.rsa = rsa
        self.temp = temp
        self.depth = depth
        self.noisy = noisy
        self.conditional_independence = conditional_independence

    def speak_step(self, listen_probs: Tensor) -> Tensor:
        """Represents a speaker step in the RSA recursion.

        Takes and returns (n_utterances, n_worlds). The input is a distribution over dim1; output over dim0."""
        utilities = (listen_probs + 1e-6).log()
        # utilities = torch.where(utilities > -INF, utilities, -INF * torch.ones_like(utilities))
        # energies = utilities - self.rsa.costs.unsqueeze(dim=1)
        # return (self.temp * energies).softmax(dim=0)
        scores = torch.exp(self.temp * (utilities - self.rsa.costs.unsqueeze(dim=-1)))

        probs = scores / scores.sum(dim=0, keepdim=True)
        return _fix_nans(probs, dim=0)

    def listen_step(self, speak_probs: Tensor, prior: Tensor) -> Tensor:
        """Represents a listener step in the RSA process.

        Takes and returns (n_utterances, n_worlds). The input is a distribution over dim0; output over dim1."""
        scores = speak_probs * prior.unsqueeze(dim=0)
        probs = scores / scores.sum(dim=1, keepdim=True)
        return _fix_nans(probs, dim=1)

    def get_listen_speak_probs(
        self,
        inferred_belief_state: Optional[Tensor] = None,
        context: List[int] = None,
    ) -> Tensor:
        """Return a conditional distribution over utterances to produce in every world.

        If context is provided, the prior is taken to be what the deepest RSA listener would have inferred from
        that context.

        Returns: listen_probs, speak_probs; a pair of (n_utterances, n_worlds)"""
        if not self.conditional_independence and context:
            # The context-dependent case, where the prior is informed by linguistic context.
            word = context[-1]
            full_prior, _ = self.get_listen_speak_probs(inferred_belief_state, context[:-1])
            prior = full_prior[word, :]
        elif inferred_belief_state is not None:
            # The case where a prior is provided, and there is no context to condition on.
            prior = inferred_belief_state
        else:
            # The case where no prior is provided, so we use the uniform distribution over worlds.
            prior = (
                _uniform(self.rsa.num_worlds)
                if inferred_belief_state is None
                else inferred_belief_state
            )

        if self.noisy:
            errors = self.rsa.errors.unsqueeze(dim=-1) if len(self.rsa.errors.shape) == 1 else self.rsa.errors
            meanings = self.rsa.truth_values * (1 - errors) + ~self.rsa.truth_values * errors
        else:
            meanings = self.rsa.truth_values

        if self.depth % 2 == 0:
            # A speaker-first RSA model under our indexing scheme.
            listen_probs = meanings
            for _ in range(self.depth // 2 + 1):
                old_listen_probs = listen_probs
                speak_probs = self.speak_step(listen_probs)
                listen_probs = self.listen_step(speak_probs, prior)
            listen_probs = old_listen_probs

        else:
            # A listener-first RSA model under our indexing scheme.
            speak_probs = meanings
            for _ in range(self.depth // 2 + 1):
                listen_probs = self.listen_step(speak_probs, prior)
                speak_probs = self.speak_step(listen_probs)

        return listen_probs, speak_probs


    def get_listen_speak_probs_wrapper(
        self,
        inferred_belief_state: Optional[Tensor] = None,
        context: List[List[int]] = None,
    ):
        """Get the speaker probabilities, handling the presence or absence of context properly"""
        if context:
            context_idxs = [self.rsa.utterances.index(c) for c in context]
        else:
            context_idxs = None
        _, speak_probs = self.get_listen_speak_probs(inferred_belief_state, context_idxs)
        return speak_probs


    def speak(
        self,
        world: int,
        inferred_belief_state: Optional[Tensor] = None,
        context: List[List[int]] = None,
    ):
        """Sample an utterance, potentially conditioned on one sentence of preceding context."""
        speak_probs = self.get_listen_speak_probs_wrapper(inferred_belief_state, context)
        dist = Categorical(speak_probs[:, world])
        utter_idx = dist.sample()
        return self.rsa.utterances[utter_idx]


    # def score_next(
    #     self,
    #     utterance: List[int],
    #     context: List[List[int]],
    #     inferred_belief_state: Optional[Tensor] = None,
    # ):
    #     """Score the likelihood a single utterance conditioned on a context"""
    #     speak_probs = self.get_listen_speak_probs_wrapper(inferred_belief_state, context)
    #     utterance_idx = self.rsa.utterances.index(utterance)
    #
    #     # if not self.conditional_independence and context:
    #     #     # The context-dependent case, where the prior is informed by linguistic context.
    #     #     word = context[-1]
    #     #     full_prior, _ = self.get_listen_speak_probs(inferred_belief_state, context[:-1])
    #     #     prior = full_prior[word, :]
    #     prob
    #
    #     log_prob = math.log(sum(speak_probs[utterance_idx, :]))
    #     return log_prob


    def score_all(
        self,
        utterances: List[List[int]],
        inferred_belief_state: Optional[Tensor] = None,
        context: List[List[int]] = None,
    ):
        """Score the likelihood a sequence of utterances, optionally conditioned on a context
        P(U) = sum_w P(U|w) P(w)
             = sum_w [ prod_i P(u_i|u_1,...,u_{i-1},w) ] P(w)
        """
        prior = None
        if not self.conditional_independence and context:
            # The context-dependent case, where the prior is informed by linguistic context.
            word = context[-1]
            full_prior, _ = self.get_listen_speak_probs(inferred_belief_state, context[:-1])
            prior = full_prior[word, :]

        if context is None:
            context = []

        speaker_probs = prior if prior else torch.ones(self.rsa.num_worlds)/self.rsa.num_worlds
        for utt in utterances:
            if len(context) > 0 and self.rsa.syntax.is_empty(context[-1]):    # If the stop token has already been provided the probability is 0
                speaker_probs = torch.zeros(self.rsa.num_worlds)
                break
            else:
                speaker_probs_utt = self.get_listen_speak_probs_wrapper(inferred_belief_state, context)
                utt_idx = self.rsa.utterances.index(utt)
                speaker_probs = torch.mul(speaker_probs, speaker_probs_utt[utt_idx])
                # log_prob += self.score_next(utt, context, inferred_belief_state)
                context.append(utt)
        prob = sum(speaker_probs)
        return prob
