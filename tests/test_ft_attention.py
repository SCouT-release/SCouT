import unittest

import torch

from src.ft_attention import (
    configure_ft_attention,
    ft_attention_parameter_names,
)


class AttentionFinetuningTest(unittest.TestCase):
    def test_only_attention_projection_weights_are_trainable(self):
        model = torch.nn.Sequential(
            torch.nn.MultiheadAttention(4, 2, batch_first=True),
            torch.nn.Linear(4, 4),
        )

        parameters = configure_ft_attention(model, verbose=False)
        trainable_names = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }

        self.assertEqual(trainable_names, {"0.in_proj_weight", "0.out_proj.weight"})
        self.assertEqual(len(parameters), 2)

    def test_attention_biases_are_optional(self):
        model = torch.nn.MultiheadAttention(4, 2, batch_first=True)

        names = ft_attention_parameter_names(model, include_bias=True)

        self.assertEqual(
            names,
            {"in_proj_weight", "out_proj.weight", "in_proj_bias", "out_proj.bias"},
        )


if __name__ == "__main__":
    unittest.main()
