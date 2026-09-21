import torch

WUDI_KEY_PATTERNS = (
    "attn.in_proj_weight",
    "attn.out_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


def _parameter_name(task_vector, key):
    prefix = "model.params."
    if not key.startswith(prefix) or not hasattr(task_vector, "param_names"):
        return key
    return task_vector.param_names[int(key[len(prefix):])]


def _target_keys(task_vectors):
    task_vector = task_vectors[0]
    vector = task_vector.vector
    keys = [
        key
        for key, value in vector.items()
        if value.ndim == 2
        and any(
            pattern in _parameter_name(task_vector, key)
            for pattern in WUDI_KEY_PATTERNS
        )
    ]
    if keys:
        return keys

    print(
        "WUDI key patterns were not found; falling back to all 2D task-vector tensors."
    )
    return [
        key
        for key, value in vector.items()
        if value.ndim == 2 and "model.params0" not in key
    ]


def _merge_tensor(key, task_vectors, args):
    values = torch.stack(
        [
            task_vector.vector[key].detach().to(args.device).float()
            for task_vector in task_vectors
        ]
    )
    merging_vector = torch.nn.Parameter(values.sum(dim=0).clone())
    optimizer = torch.optim.Adam([merging_vector], lr=args.wudi_lr, weight_decay=0.0)
    l2_norms = values.reshape(values.shape[0], -1).norm(p=2, dim=-1).square()
    l2_norms = l2_norms.clamp_min(torch.finfo(values.dtype).eps)

    for _ in range(args.wudi_steps):
        disturbing_vectors = merging_vector.unsqueeze(0) - values
        inner_product = torch.matmul(disturbing_vectors, values.transpose(1, 2))
        loss = (inner_product.square() / l2_norms[:, None, None]).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return merging_vector.detach().to(task_vectors[0].vector[key].dtype).cpu()


def merge_task_vectors(task_vectors, args):
    target_keys = set(_target_keys(task_vectors))
    print(
        f"Running WUDI-Merging on {len(target_keys)} linear-layer task-vector tensors "
        f"for {args.wudi_steps} steps."
    )
    merged_vector = {}
    for key, value in task_vectors[0].vector.items():
        if key in target_keys:
            merged_vector[key] = _merge_tensor(key, task_vectors, args)
        else:
            merged_vector[key] = torch.zeros_like(value)
    return task_vectors[0].__class__(vector=merged_vector)
