from unittest import TestCase
import torch
from torch.testing import assert_allclose

from src.rational_speech import RationalAgent, RationalDialog


def build_dialog(lang, worlds, costs):
    utterances = list(lang.keys())
    truth_values = torch.tensor([[lang[u](w) for w in worlds] for u in utterances])
    return RationalDialog(utterances, truth_values, costs)


class TestRationalAgent(TestCase):

    def setUp(self):
        torch.random.manual_seed(2)
        lang = {
            "none": lambda n: n == 0,
            "some": lambda n: n > 0,
            "all": lambda n: n == 3,
        }
        lang_costs = {
            "none": lambda n: n == 0,
            "some": lambda n: n > 0,
            "all": lambda n: n == 3,
            "null": lambda n: True,
        }
        worlds = torch.tensor([0, 1, 2, 3])
        self.quantifiers_dialog = build_dialog(lang, worlds, torch.tensor([1., 1., 1.]))
        self.quantifiers_dialog_costs = build_dialog(lang_costs, worlds, torch.tensor([1., 1., 1., 0.]))

    @torch.no_grad()
    def test_quantifiers_listen(self):
        listener = RationalAgent(self.quantifiers_dialog)
        none_belief, some_belief, all_belief = listener.listen()
        assert_allclose(none_belief, torch.tensor([1., 0., 0., 0.]))
        assert_allclose(some_belief, torch.tensor([0., .4, .4, .2]))
        assert_allclose(all_belief, torch.tensor([0., 0., 0., 1.]))

    @torch.no_grad()
    def test_quantifiers_speak_all(self):
        speaker = RationalAgent(self.quantifiers_dialog)
        speak_probs = speaker.speak()
        assert_allclose(speak_probs[:, 0], torch.tensor([1., 0., 0.]))
        assert_allclose(speak_probs[:, 1], torch.tensor([0., 1., 0.]))
        assert_allclose(speak_probs[:, 2], torch.tensor([0., 1., 0.]))
        assert_allclose(speak_probs[:, 3], torch.tensor([0., .25, .75]))

    @torch.no_grad()
    def test_sample_monologue(self):
        belief_state = torch.tensor([1., 0., 0., 0.])
        speaker = RationalAgent(self.quantifiers_dialog_costs)
        monologue = speaker.sample_monologue(belief_state)
        # The "none" here is low probability, but gets sampled with this fixed seed.
        self.assertEqual(monologue, ["none", "null", "null", "none", "null"])

    # @torch.no_grad()
    # def test_sample_dialog(self):
    #     state0 = torch.tensor([.5, .5, 0., 0.])
    #     state1 = torch.tensor([0., .5, .5, 0.])
    #     speaker = RationalAgent(self.quantifiers_dialog_costs)
    #     dialog = speaker.sample_dialog(state0, state1)
    #     breakpoint()

    # FIXME: Does temp work properly?
