import unittest

import torch

from krgh_yolo.hhca import (
    ClassificationCBAMAttention,
    ClassificationHeadHandAttention,
    CoordinateGate,
)
from krgh_yolo.krgfusion import KeyRegionGuidedFusion


class ResidualAttentionTestCase(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.x = torch.randn(2, 32, 16, 16, requires_grad=True)

    def assert_valid_residual_output(self, module):
        module.train()
        output = module(self.x)
        self.assertEqual(output.shape, self.x.shape)
        self.assertTrue(torch.isfinite(output).all())
        max_change = torch.tanh(module.scale.detach()).abs() * self.x.detach().abs().max()
        actual_change = (output.detach() - self.x.detach()).abs().max()
        self.assertLessEqual(actual_change.item(), max_change.item() + 1e-6)
        output.mean().backward()
        self.assertIsNotNone(self.x.grad)
        self.assertTrue(torch.isfinite(self.x.grad).all())


class TestKRGFusion(ResidualAttentionTestCase):
    def test_context_modes(self):
        for mode in ("local", "strip", "local_strip"):
            with self.subTest(mode=mode):
                self.x = self.x.detach().clone().requires_grad_(True)
                module = KeyRegionGuidedFusion(32, context_mode=mode)
                self.assert_valid_residual_output(module)


class TestHHCA(ResidualAttentionTestCase):
    def test_branch_modes(self):
        for mode in ("posture", "hand_object", "dual"):
            with self.subTest(mode=mode):
                self.x = self.x.detach().clone().requires_grad_(True)
                module = ClassificationHeadHandAttention(32, branch_mode=mode)
                self.assert_valid_residual_output(module)

    def test_coordinate_gate_range(self):
        gate = CoordinateGate(32)(self.x.detach())
        self.assertEqual(gate.shape, self.x.shape)
        self.assertGreaterEqual(gate.min().item(), 0.0)
        self.assertLessEqual(gate.max().item(), 1.0)

    def test_cbam_control(self):
        output = ClassificationCBAMAttention(32)(self.x.detach())
        self.assertEqual(output.shape, self.x.shape)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
