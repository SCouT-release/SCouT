import unittest

import torch

from src.PCGrad import PCGrad


class PCGradTest(unittest.TestCase):
    def test_nonconflicting_gradients_reduce_to_mean(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        pcgrad = PCGrad([parameter])

        info = pcgrad.project_and_assign(
            [[torch.tensor([1.0, 0.0])], [torch.tensor([0.0, 1.0])]]
        )

        torch.testing.assert_close(parameter.grad, torch.tensor([0.5, 0.5]))
        self.assertEqual(info["conflict_rate"], 0.0)

    def test_conflicting_gradients_are_projected(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        pcgrad = PCGrad([parameter])

        info = pcgrad.project_and_assign(
            [[torch.tensor([1.0, 0.0])], [torch.tensor([-1.0, 1.0])]]
        )

        torch.testing.assert_close(parameter.grad, torch.tensor([0.25, 0.75]))
        self.assertEqual(info["conflict_rate"], 1.0)

    def test_complete_cancellation_is_exactly_zero(self):
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        pcgrad = PCGrad([parameter])

        pcgrad.project_and_assign([[torch.tensor(0.7)], [torch.tensor(-1.3)]])

        self.assertEqual(parameter.grad.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
