import json
import os

import torch


def _run_suffix(run_name):
    return "" if not run_name else f"_{run_name}"


def uw_checkpoint_path(save_dir, run_name=None):
    return os.path.join(
        save_dir, f"hard_mtl_uw_finetuned{_run_suffix(run_name)}.pt"
    )


def uw_statistics_path(save_dir, run_name=None):
    filename = f"hard_mtl_uw_statistics{_run_suffix(run_name)}.json"
    return os.path.join(save_dir, filename)


def uw_accuracy_name(run_name=None):
    return f"hard_mtl_uw_accuracies{_run_suffix(run_name)}.json"


class UncertaintyWeighting(torch.nn.Module):
    """Learn one homoscedastic uncertainty weight per classification task."""

    def __init__(self, num_tasks):
        super().__init__()
        if num_tasks < 1:
            raise ValueError("num_tasks must be positive.")
        self.log_variances = torch.nn.Parameter(torch.zeros(num_tasks))

    def forward(self, task_loss, task_idx):
        log_variance = self.log_variances[task_idx]
        return torch.exp(-log_variance) * task_loss + 0.5 * log_variance

    @torch.no_grad()
    def statistics(self, task_names):
        if len(task_names) != self.log_variances.numel():
            raise ValueError("Expected one task name per learned uncertainty.")

        log_variances = self.log_variances.detach().cpu()
        variances = torch.exp(log_variances)
        weights = torch.exp(-log_variances)
        normalized_weights = weights / weights.mean()

        return {
            "tasks": [
                {
                    "dataset": task_name,
                    "log_variance": log_variances[idx].item(),
                    "variance": variances[idx].item(),
                    "weight": weights[idx].item(),
                    "normalized_weight": normalized_weights[idx].item(),
                }
                for idx, task_name in enumerate(task_names)
            ]
        }


def save_uw_statistics(uncertainty_weighting, task_names, save_dir, run_name=None):
    path = uw_statistics_path(save_dir, run_name)
    with open(path, "w") as f:
        json.dump(uncertainty_weighting.statistics(task_names), f, indent=2)
    print(f"Saving learned uncertainty statistics to {path}")
    return path
