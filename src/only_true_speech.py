"""Simple agent that just randomly samples a sentence that is true."""

from typing import List
from torch import Tensor
from torch.distributions import Categorical
import random

from src.rational_speech import RationalSpeechActs

class OnlyTrueAgent:

    """Randomly say a true thing."""

    def __init__(self, rsa: RationalSpeechActs):
        self.rsa = rsa
    
    def sample_monologue(self,
                         belief_state: Tensor,
                         length: int = 5,
                         update_prior: bool = True,
                        ) -> List[str]:
        for _ in range(length):
            world = Categorical(belief_state).sample()
            true_utterances = [u for u, t in zip(self.rsa.utterances, self.rsa.truth_values[:, world]) if t]
            yield random.choice(true_utterances)
