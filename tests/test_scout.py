import copy
import importlib.util
import sys
import unittest
from unittest import mock

import torch

from src.args import parse_arguments
from src.soft_joint import soft_joint_loss


class ToySpecialist(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.Linear(4, 2),
        )


class ScoutTest(unittest.TestCase):
    def test_public_cli_name_maps_to_legacy_checkpoint_mode(self):
        with mock.patch.object(sys, "argv", ["test", "--finetuning-mode", "scout"]):
            args = parse_arguments()

        self.assertEqual(args.finetuning_mode, "soft_joint")

    def test_zero_coupling_has_zero_loss(self):
        models = [ToySpecialist(), ToySpecialist()]
        pretrained_state = {
            name: torch.zeros_like(value)
            for name, value in models[0].image_encoder.state_dict().items()
        }

        loss = soft_joint_loss(models, pretrained_state, coupling_lambda=0.0)

        self.assertEqual(loss.item(), 0.0)

    def test_sharded_gradient_matches_scout_objective(self):
        if importlib.util.find_spec("open_clip") is None:
            self.skipTest("open_clip is required to import the training entry point")

        from src.soft_joint_finetune import _add_sharded_coupling_gradients

        torch.manual_seed(7)
        models = [ToySpecialist() for _ in range(3)]
        reference_models = copy.deepcopy(models)
        coupling_lambda = 0.4
        pretrained_state = {
            name: torch.zeros_like(value)
            for name, value in reference_models[0].image_encoder.state_dict().items()
        }
        reference_loss = soft_joint_loss(
            reference_models,
            pretrained_state,
            coupling_lambda,
        )
        reference_loss.backward()

        computed_loss = _add_sharded_coupling_gradients(
            models,
            total_tasks=len(models),
            coupling_lambda=coupling_lambda,
        )

        torch.testing.assert_close(torch.tensor(computed_loss), reference_loss.detach())
        for model, reference_model in zip(models, reference_models):
            for parameter, reference_parameter in zip(
                model.image_encoder.parameters(),
                reference_model.image_encoder.parameters(),
            ):
                torch.testing.assert_close(parameter.grad, reference_parameter.grad)


if __name__ == "__main__":
    unittest.main()
