import json
import os
from typing import Dict, Iterable, Tuple

import torch

from src.datasets.common import maybe_dictionarize

FisherMetric = Dict[str, torch.Tensor]


def rowwise_grad_square(grad: torch.Tensor) -> torch.Tensor:
    """Reduce a per-example gradient square to a row-wise diagonal metric."""
    grad_sq = grad.detach().float().square()
    if grad_sq.ndim <= 1:
        return grad_sq
    reduce_dims = tuple(range(1, grad_sq.ndim))
    return grad_sq.mean(dim=reduce_dims, keepdim=True)


def diagonal_grad_square(grad: torch.Tensor) -> torch.Tensor:
    """Return the element-wise diagonal empirical Fisher contribution."""
    return grad.detach().float().square()


def expanded_metric_sum(metric: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    expansion_factor = param.numel() / metric.numel()
    return metric.float().sum() * expansion_factor


def expanded_metric_mean(
    fisher: FisherMetric,
    named_params: Dict[str, torch.nn.Parameter],
) -> torch.Tensor:
    total_sum = torch.zeros((), device=next(iter(fisher.values())).device)
    total_count = 0
    for name, metric in fisher.items():
        if name not in named_params:
            raise ValueError(f"Fisher metric has unknown parameter {name}.")
        total_sum = total_sum + expanded_metric_sum(metric, named_params[name])
        total_count += named_params[name].numel()
    if total_count == 0:
        raise ValueError("Cannot normalize Fisher for an empty parameter set.")
    return total_sum / total_count


@torch.no_grad()
def normalize_fisher_metric(
    fisher: FisherMetric,
    named_params: Dict[str, torch.nn.Parameter],
    identity_mix: float,
    fisher_min: float,
    fisher_max: float,
) -> Tuple[FisherMetric, dict]:
    mean_before = expanded_metric_mean(fisher, named_params)
    if not torch.isfinite(mean_before) or mean_before.item() <= 0:
        raise ValueError(f"Invalid Fisher mean before normalization: {mean_before.item()}.")

    normalized = {}
    for name, metric in fisher.items():
        normalized_metric = metric.float() / mean_before
        normalized_metric = identity_mix + (1.0 - identity_mix) * normalized_metric
        normalized_metric = normalized_metric.clamp(min=fisher_min, max=fisher_max)
        if not torch.isfinite(normalized_metric).all():
            raise ValueError(f"Nonfinite Fisher metric after normalization for {name}.")
        normalized[name] = normalized_metric.detach().cpu()

    mean_after = expanded_metric_mean(normalized, named_params)
    summary = {
        "min": min(metric.min().item() for metric in normalized.values()),
        "max": max(metric.max().item() for metric in normalized.values()),
        "expanded_mean_before": mean_before.item(),
        "expanded_mean_after": mean_after.item(),
    }
    return normalized, summary


def estimate_empirical_fisher(
    model: torch.nn.Module,
    named_params: Dict[str, torch.nn.Parameter],
    dataloader: Iterable,
    device: str,
    num_samples: int,
    identity_mix: float,
    fisher_min: float,
    fisher_max: float,
    metric_fn=rowwise_grad_square,
) -> Tuple[FisherMetric, int, dict]:
    """Estimate a normalized empirical Fisher from validation examples."""
    if num_samples <= 0:
        raise ValueError("--fisher-num-samples must be positive.")

    all_names = list(named_params.keys())
    all_params = [named_params[name] for name in all_names]
    param_names = None
    params = None
    fisher_sum = None
    loss_fn = torch.nn.CrossEntropyLoss()
    was_training = model.training
    model.eval()

    samples_used = 0
    for batch in dataloader:
        batch = maybe_dictionarize(batch)
        inputs = batch["images"].to(device)
        labels = batch["labels"].to(device)

        for sample_idx in range(inputs.shape[0]):
            if samples_used >= num_samples:
                break

            logits = model(inputs[sample_idx : sample_idx + 1])
            loss = loss_fn(logits, labels[sample_idx : sample_idx + 1])
            if params is None:
                grads = torch.autograd.grad(
                    loss,
                    all_params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                used = [
                    (name, param, grad)
                    for name, param, grad in zip(all_names, all_params, grads)
                    if grad is not None
                ]
                if not used:
                    raise ValueError("No encoder parameters were used by image forward.")
                param_names = [name for name, _, _ in used]
                params = [param for _, param, _ in used]
                grads = [grad for _, _, grad in used]
                fisher_sum = {
                    name: torch.zeros_like(metric_fn(param), device=device)
                    for name, param in zip(param_names, params)
                }
            else:
                grads = torch.autograd.grad(
                    loss,
                    params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )

            for name, grad in zip(param_names, grads):
                fisher_sum[name].add_(metric_fn(grad).to(device))
            samples_used += 1

        if samples_used >= num_samples:
            break

    if samples_used == 0:
        raise ValueError("Fisher calibration did not see any validation examples.")

    fisher = {
        name: metric_sum / samples_used
        for name, metric_sum in fisher_sum.items()
    }
    normalized, summary = normalize_fisher_metric(
        fisher,
        {name: named_params[name] for name in param_names},
        identity_mix=identity_mix,
        fisher_min=fisher_min,
        fisher_max=fisher_max,
    )
    summary["num_samples"] = samples_used
    summary["num_used_parameters"] = len(param_names)
    summary["num_skipped_parameters"] = len(all_names) - len(param_names)

    model.train(was_training)
    return normalized, samples_used, summary


def save_fisher_metric(path: str, fisher: FisherMetric, metadata: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({name: metric.cpu() for name, metric in fisher.items()}, path)

    metadata_path = fisher_metadata_path(path)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)


def load_fisher_metric(path: str) -> FisherMetric:
    fisher = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(fisher, dict):
        raise ValueError(f"Expected Fisher metric dictionary in {path}.")
    return {name: metric.detach().cpu() for name, metric in fisher.items()}


def fisher_metadata_path(path: str) -> str:
    root, _ = os.path.splitext(path)
    return f"{root}_metadata.json"


def fisher_path(args, train_dataset):
    run_prefix = "" if not args.run_name else f"{args.run_name}_"
    return os.path.join(
        args.save,
        train_dataset,
        f"{run_prefix}{args.curvature_mode}_fisher.pt",
    )


def named_trainable_encoder_params(model):
    return {
        name: param
        for name, param in model.image_encoder.named_parameters()
        if param.requires_grad
    }


def validate_matching_named_params(states):
    names = list(states[0].named_encoder_params.keys())
    shapes = {
        name: tuple(param.shape)
        for name, param in states[0].named_encoder_params.items()
    }
    for state in states[1:]:
        if list(state.named_encoder_params.keys()) != names:
            raise ValueError("Curvature-aware task encoders have different parameters.")
        for name, param in state.named_encoder_params.items():
            if tuple(param.shape) != shapes[name]:
                raise ValueError(f"Parameter {name} has mismatched shapes.")
    return names


def _task_weights(num_tasks, device, task_weights=None):
    if task_weights is None:
        return torch.full((num_tasks,), 1.0 / num_tasks, device=device)
    weights = torch.as_tensor(task_weights, device=device, dtype=torch.float32)
    if weights.shape != (num_tasks,):
        raise ValueError(f"Expected {num_tasks} task weights.")
    if (weights < 0).any() or not torch.isclose(weights.sum(), torch.ones((), device=device)):
        raise ValueError("Task weights must be non-negative and sum to 1.")
    return weights


def _metric(state, name, param):
    metric = state.fisher_metric[name].to(device=param.device, dtype=param.dtype)
    try:
        torch.broadcast_shapes(tuple(metric.shape), tuple(param.shape))
    except RuntimeError as exc:
        raise ValueError(f"Fisher for {state.train_dataset}:{name} cannot broadcast.") from exc
    return metric


@torch.no_grad()
def add_curvature_coupling_gradients_(
    states,
    parameter_names,
    pretrained_params,
    coupling_lambda,
    task_weights=None,
):
    """Add gradient of lambda/2 * sum_k ||theta_m - theta_k||^2_{D_k}."""
    weights = _task_weights(
        len(states),
        next(iter(pretrained_params.values())).device,
        task_weights,
    )
    value = torch.zeros((), device=weights.device)
    for name in parameter_names:
        params = [state.named_encoder_params[name] for state in states]
        metrics = [_metric(state, name, params[0]) for state in states]
        merged = pretrained_params[name].to(params[0]) * (1.0 - weights.sum().item())
        for weight, param in zip(weights, params):
            merged.add_(param, alpha=weight.item())

        shared = torch.zeros_like(merged)
        for param, metric in zip(params, metrics):
            residual = merged - param
            shared.add_(metric * residual)
            value = value + (metric * residual.square()).sum()

        for weight, param, metric in zip(weights, params, metrics):
            grad = weight.item() * shared + metric * (param - merged)
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            param.grad.add_(grad, alpha=coupling_lambda)
    return (0.5 * coupling_lambda * value).item()


@torch.no_grad()
def compute_curvature_distance(states, parameter_names, pretrained_params, task_weights=None):
    weights = _task_weights(
        len(states),
        next(iter(pretrained_params.values())).device,
        task_weights,
    )
    distance = torch.zeros((), device=weights.device)
    for name in parameter_names:
        params = [state.named_encoder_params[name] for state in states]
        merged = pretrained_params[name].to(params[0]) * (1.0 - weights.sum().item())
        for weight, param in zip(weights, params):
            merged.add_(param, alpha=weight.item())
        for state, param in zip(states, params):
            distance = distance + (
                _metric(state, name, param) * (param - merged).square()
            ).sum()
    return (distance / len(states)).item()


# Fisher-weighted model merging.


def _validation_dataset_name(dataset_name):
    return dataset_name if dataset_name.endswith("Val") else f"{dataset_name}Val"


def _validate_fisher_merge_args(args):
    if args.fisher_batch_size <= 0:
        raise ValueError("--fisher-batch-size must be positive.")
    if args.fisher_floor <= 0:
        raise ValueError("--fisher-floor must be positive.")
    if not 0.0 <= args.fisher_identity_mix <= 1.0:
        raise ValueError("--fisher-identity-mix must be in the interval [0, 1].")
    if not 0.0 <= args.fisher_min <= args.fisher_max:
        raise ValueError("--fisher-min and --fisher-max must satisfy 0 <= min <= max.")


def _fisher_validation_loader(dataset_name, image_encoder, args):
    from src.datasets.common import get_dataloader
    from src.datasets.registry import get_dataset

    dataset = get_dataset(
        _validation_dataset_name(dataset_name),
        image_encoder.val_preprocess,
        location=args.data_location,
        batch_size=args.fisher_batch_size,
    )
    return get_dataloader(dataset, is_train=False, args=args, image_encoder=None)


def _fisher_task_model(task_vector, pretrained_checkpoint, dataset_name, args):
    from src.heads import get_classification_head
    from src.modeling import ImageClassifier

    image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
    model = ImageClassifier(
        image_encoder,
        get_classification_head(args, dataset_name),
    )
    model.freeze_head()
    return model.to(args.device)


def _accumulate_fisher_weighted_state(
    numerators,
    denominators,
    theta_sums,
    model,
    fisher,
    keys,
):
    state_dict = model.image_encoder.state_dict()
    for key in keys:
        theta = state_dict[key].detach().cpu().float()
        theta_sums[key].add_(theta)
        if key not in fisher:
            continue

        metric = fisher[key].float()
        try:
            metric = torch.broadcast_to(metric, theta.shape)
        except RuntimeError as exc:
            raise ValueError(f"Fisher for {key} cannot broadcast to parameter.") from exc
        numerators[key].add_(metric * theta)
        denominators[key].add_(metric)


def merge_task_vectors(task_vectors, pretrained_checkpoint, eval_datasets, args):
    """Merge task vectors with in-memory diagonal empirical Fisher weighting."""
    _validate_fisher_merge_args(args)
    print(
        f"Running Fisher merging with {args.fisher_num_samples} validation samples "
        "per task."
    )
    pretrained = task_vectors[0].apply_to(pretrained_checkpoint, scaling_coef=0.0)
    pretrained_state = {
        key: value.detach().cpu().float()
        for key, value in pretrained.state_dict().items()
        if key in task_vectors[0].vector
    }
    keys = list(pretrained_state.keys())
    numerators = {key: torch.zeros_like(value) for key, value in pretrained_state.items()}
    denominators = {key: torch.zeros_like(value) for key, value in pretrained_state.items()}
    theta_sums = {key: torch.zeros_like(value) for key, value in pretrained_state.items()}

    for dataset_name, task_vector in zip(eval_datasets, task_vectors):
        model = _fisher_task_model(task_vector, pretrained_checkpoint, dataset_name, args)
        named_params = named_trainable_encoder_params(model)
        fisher, samples_used, summary = estimate_empirical_fisher(
            model,
            named_params,
            _fisher_validation_loader(dataset_name, model.image_encoder, args),
            device=args.device,
            num_samples=args.fisher_num_samples,
            identity_mix=args.fisher_identity_mix,
            fisher_min=args.fisher_min,
            fisher_max=args.fisher_max,
            metric_fn=diagonal_grad_square,
        )
        _accumulate_fisher_weighted_state(
            numerators,
            denominators,
            theta_sums,
            model,
            fisher,
            keys,
        )
        print(
            f"Fisher {dataset_name}: samples={samples_used} "
            f"mean={summary['expanded_mean_after']:.4f}",
            flush=True,
        )
        del model, fisher

    merged_vector = {}
    num_tasks = len(task_vectors)
    for key, pretrained_value in pretrained_state.items():
        has_fisher = denominators[key] > 0
        fisher_theta = numerators[key] / denominators[key].clamp_min(args.fisher_floor)
        average_theta = theta_sums[key] / num_tasks
        merged_theta = torch.where(has_fisher, fisher_theta, average_theta)
        merged_vector[key] = (
            merged_theta - pretrained_value
        ).to(task_vectors[0].vector[key].dtype)

    return task_vectors[0].__class__(vector=merged_vector)
