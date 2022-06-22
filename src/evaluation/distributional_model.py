from typing import Optional, List
from src.rational_speech import RationalAgent
import math
from src.powerset.serialize import from_string
from collections import Counter
import numpy as np

class DistributionalModel():

    def __init__(self, empty):
        self.empty = empty
        self.epsilon = 1e-20

    def score(self, sentence: str, context: Optional[List[str]] = None) -> float:
        pass

    def logscore(self, sentence, context):
        return math.log(self.score(sentence, context))

    def concat(self, s1: str, s2: str) -> str:
        s1 = s1.removesuffix(self.empty).strip() if s1 != self.empty else s1
        s2 = s2.removesuffix(self.empty).strip() if s2 != self.empty else s2
        return s1 + " " + s2

    def test_uniform_true(self, premise, hypothesis):
        """[[x]] ⊆ [[y]] <==> p(xy) = p(xx)"""
        xx = self.concat(premise, premise)
        xy = self.concat(premise, hypothesis)
        lhs = self.score(xx)
        rhs = self.score(xy)
        return abs(lhs - rhs)

    def test_gricean(self, premise, hypothesis, log_ratio=False):
        """[[x]] ⊆ [[y]] <==> p(xy)/p(xT) = p(yy)/p(yT)"""
        xy = self.concat(premise, hypothesis)
        xT = self.concat(premise, self.empty)
        yy = self.concat(hypothesis, hypothesis)
        yT = self.concat(hypothesis, self.empty)
        lhs = self.score(xy) / self.score(xT)
        rhs = self.score(yy) / self.score(yT)
        if log_ratio:
            return abs(math.log(lhs) - math.log(rhs))
        else:
            return abs(lhs - rhs)


class NgramModel(DistributionalModel):

    def __init__(self, empty, lm):
        super().__init__(empty)
        self.lm = lm

    @staticmethod
    def train_lm(order, train_path=None, text=None):
        from nltk.lm.preprocessing import padded_everygram_pipeline
        from nltk.lm import MLE
        if text is None:
            if train_path is None:
                raise ValueError("One of train_path or text must be provided.")
            text = [line.split() for line in open(train_path)]
        train, vocab = padded_everygram_pipeline(order, text)
        lm = MLE(order)
        lm.fit(train, vocab)
        return lm

    def score(self, sentence: str, context: Optional[List[str]] = None) -> float:
        """Score a sentence. Do not include padding, but may include model's own EOS token: 1^|w|"""
        p = 1
        context = ["<s>"] * (self.lm.order - 1)
        for word in sentence.split():
            p *= self.lm.score(word, context)
            context = context[1:] + [word]
        return p if p != 0.0 else self.epsilon


class RSAModel(DistributionalModel):

    def __init__(self, empty, rational_agent: RationalAgent):
        super().__init__(empty)
        self.rational_agent = rational_agent

    def score(self, sentence: str, context: Optional[List[str]] = None) -> float:
        sentence = from_string(sentence)
        p = self.rational_agent.score(utterances=sentence, context=context)
        return p if p != 0.0 else self.epsilon


class TextFrequency(DistributionalModel):

    def __init__(self, empty, lm=None, text=None):
        super().__init__(empty)
        if lm is not None:
            self.lm = lm
        elif text is not None:
            self.lm = self.train_lm(text=text)
        else:
            raise ValueError("Either lm or train_path argument must be supplied.")

    def train_lm(self, train_path=None, text=None):
        if text is None:
            if train_path is None:
                raise ValueError("One of train_path or text must be provided.")
            text = [line.removesuffix(self.empty) for line in open(train_path)]
        counts = Counter(text)
        total = len(text)
        lm = {k: counts[k] / total for k in counts.keys()}
        self.lm = lm
        return lm

    def score(self, sentence: str, context: Optional[List[str]] = None) -> float:
        """Score a sentence. Do not include padding, but may include model's own EOS token: 1^|w|"""
        if context is not None:
            raise ValueError("This model can't predict conditional probabilities.")
        if sentence in self.lm:
            p = self.lm[sentence]
            return p if p != 0.0 else self.epsilon
        else:
            return self.epsilon


# def test_entailment_uniform_true(sents1, sents2, labels):
#     """[[x]] ⊆ [[y]] <==> p(xy) = p(xx)"""
#     xy = [f"{x} {y}" for x, y in zip(sents1, sents2)]
#     xx = [f"{x} {x}" for x in sents1]
#     for model_name in models:
#         model_path = os.path.join(args.model_dir, model_name)
#         predictor = get_predictor(model_path)
#         lhs = [sum(score(s, predictor)[:-1]).item() for s in xy]
#         rhs = [sum(score(s, predictor)[:-1]).item() for s in xx]
#         p_diff = [abs(a - b) for a, b in zip(lhs, rhs)]
#         auc_score = auc(p_diff, labels)
#         scatterplot(lhs, rhs, labels, f"{model_name}_uniform", auc=auc_score)
#         kdeplot(p_diff, labels, f"{model_name}_uniform", auc=auc_score)
