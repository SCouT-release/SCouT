import unittest

import torch

from src.mergopt import MergOPT, parse_mergopt_alphas


class MergOPTTest(unittest.TestCase):
    def test_step_uses_perturbed_gradient_and_restored_parameters(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = MergOPT(
            [parameter],
            torch.optim.SGD,
            lr=0.1,
            weight_decay=0.0,
            mu=1.0,
            b=0.0,
            k_max=1,
            alphas=(2.0,),
        )

        def closure():
            optimizer.zero_grad()
            self.assertEqual(parameter.item(), 2.0)
            loss = parameter.square()
            loss.backward()
            return loss.detach()

        loss = optimizer.step(closure)

        self.assertEqual(loss.item(), 4.0)
        self.assertAlmostEqual(parameter.item(), 0.6, places=6)

    def test_without_closure_matches_base_optimizer(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = MergOPT(
            [parameter],
            torch.optim.SGD,
            lr=0.1,
            weight_decay=0.0,
        )

        parameter.square().backward()
        optimizer.step()

        self.assertAlmostEqual(parameter.item(), 0.8, places=6)

    def test_parse_alphas(self):
        self.assertEqual(parse_mergopt_alphas("0.1,0.2,0.6"), (0.1, 0.2, 0.6))


if __name__ == "__main__":
    unittest.main()
