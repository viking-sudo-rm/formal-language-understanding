from unittest import TestCase
import torch
from torch.testing import assert_allclose

from src.rational_speech import RationalAgent, RationalSpeechActs


def build_dialog(lang, worlds, costs):
    utterances = list(lang.keys())
    truth_values = torch.tensor([[lang[u](w) for w in worlds] for u in utterances])
    return RationalSpeechActs(utterances, truth_values, costs)


class TestRationalAgent(TestCase):

    def setUp(self):
        torch.random.manual_seed(0)
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
        self.rsa = build_dialog(lang, worlds, torch.tensor([1., 1., 1.]))
        self.rsa_costs = build_dialog(lang_costs, worlds, torch.tensor([1., 1., 1., 0.]))

    # @torch.no_grad()
    # def test_quantifiers_listen(self):
    #     listener = RationalAgent(self.rsa)
    #     none_belief, some_belief, all_belief = listener.listen()
    #     assert_allclose(none_belief, torch.tensor([1., 0., 0., 0.]))
    #     assert_allclose(some_belief, torch.tensor([0., .4, .4, .2]))
    #     assert_allclose(all_belief, torch.tensor([0., 0., 0., 1.]))

    @torch.no_grad()
    def test_speak0(self):
        speaker = RationalAgent(self.rsa, depth=0)
        _, speak_probs = speaker.get_listen_speak_probs()
        assert_allclose(speak_probs[:, 0], torch.tensor([1., 0., 0.]))
        assert_allclose(speak_probs[:, 1], torch.tensor([0., 1., 0.]))
        assert_allclose(speak_probs[:, 2], torch.tensor([0., 1., 0.]))
        assert_allclose(speak_probs[:, 3], torch.tensor([0., .5, .5]))

    @torch.no_grad()
    def test_speak1(self):
        speaker = RationalAgent(self.rsa, depth=1)
        _, speak_probs = speaker.get_listen_speak_probs()
        assert_allclose(speak_probs[:, 0], torch.tensor([1., 0., 0.]))
        assert_allclose(speak_probs[:, 1], torch.tensor([0., 1., 0.]))
        assert_allclose(speak_probs[:, 2], torch.tensor([0., 1., 0.]))
        assert_allclose(speak_probs[:, 3], torch.tensor([0., .25, .75]))

    @torch.no_grad()
    def test_speak2(self):
        speaker = RationalAgent(self.rsa, depth=2)
        _, speak_probs = speaker.get_listen_speak_probs()
        assert_allclose(speak_probs[:, 0], torch.tensor([1., 0., 0.]))
        assert_allclose(speak_probs[:, 1], torch.tensor([0., 1., 0.]))
        assert_allclose(speak_probs[:, 2], torch.tensor([0., 1., 0.]))
        assert_allclose(speak_probs[:, 3], torch.tensor([0., .16666666, .8333333]))

    @torch.no_grad()
    def test_speak2_rational(self):
        # Using a temperature of 1000 causes errors. Overflow, maybe?
        speaker = RationalAgent(self.rsa, depth=2, temp=100)
        _, speak_probs = speaker.get_listen_speak_probs()
        assert_allclose(speak_probs[:, 0], torch.tensor([1., 0., 0.]))
        assert_allclose(speak_probs[:, 1], torch.tensor([0., 1., 0.]))
        assert_allclose(speak_probs[:, 2], torch.tensor([0., 1., 0.]))
        assert_allclose(speak_probs[:, 3], torch.tensor([0., 0., 1.]))

    @torch.no_grad()
    def test_speak1_rational(self):
        speaker = RationalAgent(self.rsa, depth=1)
        listen_probs, speak_probs = speaker.get_listen_speak_probs(context=[2])
        assert_allclose(listen_probs[0, :], torch.tensor([0., 0., 0., 0.]))
        assert_allclose(listen_probs[1, :], torch.tensor([0., 0., 0., 1.]))
        assert_allclose(listen_probs[2, :], torch.tensor([0., 0., 0., 1.]))
        
        assert_allclose(speak_probs[:, 0], torch.tensor([0., 0., 0.]))
        assert_allclose(speak_probs[:, 1], torch.tensor([0., 0., 0.]))
        assert_allclose(speak_probs[:, 2], torch.tensor([0., 0., 0.]))
        assert_allclose(speak_probs[:, 3], torch.tensor([0., .5, .5]))
