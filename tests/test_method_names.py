import unittest

from src.hard_mtl import hard_mtl_accuracy_name, hard_mtl_checkpoint_name
from src.pcgrad import pcgrad_accuracy_name, pcgrad_checkpoint_path
from src.result_names import (
    finetuned_checkpoint_name,
    merge_result_name,
    single_task_accuracy_name,
)
from src.scout import scout_accuracy_name, scout_checkpoint_name, scout_merge_name
from src.uncertainty_weighting import uw_accuracy_name, uw_checkpoint_path


class MethodNamesTest(unittest.TestCase):
    def test_specialist_method_names_match_paper(self):
        self.assertEqual(
            finetuned_checkpoint_name("independent_ft", "seed0"),
            "independent_ft_finetuned_seed0.pt",
        )
        self.assertEqual(
            finetuned_checkpoint_name("ftts"),
            "ftts_finetuned.pt",
        )
        self.assertEqual(
            single_task_accuracy_name("ft_attention"),
            "ft_attention_accuracies.json",
        )
        self.assertEqual(
            merge_result_name("mergopt"),
            "mergopt_merge.json",
        )

    def test_scout_names_match_paper(self):
        self.assertEqual(
            scout_checkpoint_name(1, 0.5, run_name="seed0"),
            "scout_finetuned_seed0_1_0.5.pt",
        )
        self.assertEqual(
            scout_accuracy_name(1, 0.5),
            "scout_accuracies_1_0.5.json",
        )
        self.assertEqual(
            scout_merge_name(1, 0.5),
            "scout_merge_1_0.5.json",
        )

    def test_hard_mtl_names_match_paper(self):
        self.assertEqual(hard_mtl_checkpoint_name(), "hard_mtl_finetuned.pt")
        self.assertEqual(hard_mtl_accuracy_name(), "hard_mtl_accuracies.json")
        self.assertTrue(uw_checkpoint_path("out").endswith("hard_mtl_uw_finetuned.pt"))
        self.assertEqual(uw_accuracy_name(), "hard_mtl_uw_accuracies.json")
        self.assertTrue(
            pcgrad_checkpoint_path("out").endswith("hard_mtl_pcgrad_finetuned.pt")
        )
        self.assertEqual(
            pcgrad_accuracy_name(),
            "hard_mtl_pcgrad_accuracies.json",
        )


if __name__ == "__main__":
    unittest.main()
