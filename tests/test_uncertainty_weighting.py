import unittest

import torch

from src.uncertainty_weighting import UncertaintyWeighting


class UncertaintyWeightingTest(unittest.TestCase):
    def test_zero_initialization_matches_unweighted_loss(self):
        weighting = UncertaintyWeighting(2)
        losses = [torch.tensor(2.0), torch.tensor(4.0)]

        objective = sum(weighting(loss, index) for index, loss in enumerate(losses)) / 2

        self.assertEqual(objective.item(), 3.0)

    def test_classification_objective_and_gradients(self):
        weighting = UncertaintyWeighting(1)
        loss = torch.tensor(2.0, requires_grad=True)

        objective = weighting(loss, 0)
        objective.backward()

        self.assertEqual(objective.item(), 2.0)
        self.assertEqual(loss.grad.item(), 1.0)
        self.assertEqual(weighting.log_variances.grad.item(), -1.5)

    def test_statistics_align_with_task_names(self):
        weighting = UncertaintyWeighting(2)
        statistics = weighting.statistics(["TaskA", "TaskB"])

        self.assertEqual(
            [task["dataset"] for task in statistics["tasks"]],
            ["TaskA", "TaskB"],
        )


if __name__ == "__main__":
    unittest.main()
