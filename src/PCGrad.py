import os

import torch


def _run_suffix(run_name):
    return "" if not run_name else f"_{run_name}"


def pcgrad_checkpoint_path(save_dir, run_name=None):
    return os.path.join(save_dir, f"pcgrad_finetuned{_run_suffix(run_name)}.pt")


def pcgrad_accuracy_name(run_name=None):
    return f"pcgrad_ft_accuracies{_run_suffix(run_name)}.json"


class PCGrad:
    """Project conflicting task gradients before an optimizer step."""

    def __init__(self, params, eps=1e-12):
        self.params = list(params)
        self.eps = eps
        if not self.params:
            raise ValueError("PCGrad requires at least one trainable parameter.")

    @torch.no_grad()
    def capture(self):
        return [
            torch.zeros_like(param) if param.grad is None else param.grad.detach().clone()
            for param in self.params
        ]

    @staticmethod
    def _dot(left, right):
        return sum((x * y).sum() for x, y in zip(left, right))

    @torch.no_grad()
    def project_and_assign(self, task_gradients):
        if not task_gradients:
            raise ValueError("PCGrad requires at least one task gradient.")
        if any(len(gradients) != len(self.params) for gradients in task_gradients):
            raise ValueError("Each task must provide one gradient per parameter.")

        num_tasks = len(task_gradients)
        combined = [torch.zeros_like(param) for param in self.params]
        task_norms = [self._dot(gradients, gradients) for gradients in task_gradients]
        conflicts = 0
        comparisons = num_tasks * (num_tasks - 1)

        for task_idx, gradients in enumerate(task_gradients):
            projected = [gradient.clone() for gradient in gradients]
            for other_idx in torch.randperm(num_tasks).tolist():
                if other_idx == task_idx:
                    continue
                other = task_gradients[other_idx]
                dot = self._dot(projected, other)
                dot_value = dot.item()
                norm_sq = task_norms[other_idx].item()
                if dot_value < 0 and norm_sq > self.eps:
                    scale = dot_value / norm_sq
                    for gradient, other_gradient in zip(projected, other):
                        gradient.add_(other_gradient, alpha=-scale)
                    conflicts += 1

            projected_norm = self._dot(projected, projected).item()
            if projected_norm <= self.eps * task_norms[task_idx].item():
                for gradient in projected:
                    gradient.zero_()

            for total, gradient in zip(combined, projected):
                total.add_(gradient)

        norm_sq = torch.zeros((), device=combined[0].device)
        for param, gradient in zip(self.params, combined):
            gradient.div_(num_tasks)
            param.grad = gradient
            norm_sq.add_(gradient.float().square().sum())

        return {
            "conflict_rate": conflicts / comparisons if comparisons else 0.0,
            "gradient_norm": norm_sq.sqrt().item(),
        }
