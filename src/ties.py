import torch


def _merge_keys(task_vector):
    return [key for key in task_vector.vector if not key.startswith("model.params0.")]


def _topk_threshold(task_vector, keys, keep_ratio, device):
    values = torch.cat(
        [
            task_vector.vector[key].detach().abs().reshape(-1).to(device).float()
            for key in keys
        ]
    )
    k = max(1, int(values.numel() * keep_ratio))
    return torch.topk(values, k).values[-1]


def _trim(values, thresholds):
    view_shape = (thresholds.shape[0],) + (1,) * (values.ndim - 1)
    thresholds = thresholds.view(view_shape)
    return torch.where(values.abs() >= thresholds, values, torch.zeros_like(values))


def _disjoint_mean(values, signs):
    signs = signs.unsqueeze(0)
    keep = ((signs > 0) & (values > 0)) | ((signs < 0) & (values < 0))
    selected = values * keep
    counts = keep.sum(dim=0).clamp_min(1)
    return selected.sum(dim=0) / counts


def _merge_tensor(key, task_vectors, thresholds, args):
    values = torch.stack(
        [
            task_vector.vector[key].detach().to(args.device).float()
            for task_vector in task_vectors
        ]
    )
    values = _trim(values, thresholds)
    signs = values.sum(dim=0).sign()
    merged = _disjoint_mean(values, signs)
    return merged.detach().to(task_vectors[0].vector[key].dtype).cpu()


def merge_task_vectors(task_vectors, args):
    keep_ratio = args.ties_trim_ratio
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("--ties-trim-ratio must be in the interval (0, 1].")

    print(f"Running TIES-Merging with top-{100 * keep_ratio:.1f}% trimming.")
    merge_keys = _merge_keys(task_vectors[0])
    merge_key_set = set(merge_keys)
    thresholds = torch.stack(
        [
            _topk_threshold(task_vector, merge_keys, keep_ratio, args.device)
            for task_vector in task_vectors
        ]
    )

    with torch.no_grad():
        merged_vector = {
            key: (
                _merge_tensor(key, task_vectors, thresholds, args)
                if key in merge_key_set
                else torch.zeros_like(value)
            )
            for key, value in task_vectors[0].vector.items()
        }
    return task_vectors[0].__class__(vector=merged_vector)
