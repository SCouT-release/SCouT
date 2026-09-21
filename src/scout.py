import torch


def format_coupling_value(value):
    return f"{value:g}"


def _run_prefix(run_name):
    return "" if not run_name else f"{run_name}_"


def scout_checkpoint_name(coupling_tau, coupling_lambda, step=None, run_name=None):
    tau = format_coupling_value(coupling_tau)
    lam = format_coupling_value(coupling_lambda)
    run_prefix = _run_prefix(run_name)
    if step is None:
        return f"scout_finetuned_{run_prefix}{tau}_{lam}.pt"
    return f"scout_checkpoint_{run_prefix}{tau}_{lam}_{step}.pt"


def scout_accuracy_name(coupling_tau, coupling_lambda, run_name=None):
    tau = format_coupling_value(coupling_tau)
    lam = format_coupling_value(coupling_lambda)
    run_prefix = _run_prefix(run_name)
    return f"scout_accuracies_{run_prefix}{tau}_{lam}.json"


def scout_merge_name(coupling_tau, coupling_lambda, run_name=None):
    tau = format_coupling_value(coupling_tau)
    lam = format_coupling_value(coupling_lambda)
    run_prefix = _run_prefix(run_name)
    return f"scout_merge_{run_prefix}{tau}_{lam}.json"


def scout_adamerging_name(coupling_tau, coupling_lambda, run_name=None):
    tau = format_coupling_value(coupling_tau)
    lam = format_coupling_value(coupling_lambda)
    run_prefix = _run_prefix(run_name)
    return f"scout_adamerging_{run_prefix}{tau}_{lam}.json"


def scout_negation_name(coupling_tau, coupling_lambda, run_name=None):
    tau = format_coupling_value(coupling_tau)
    lam = format_coupling_value(coupling_lambda)
    run_prefix = _run_prefix(run_name)
    return f"scout_negations_{run_prefix}{tau}_{lam}.json"


def scout_distance_name(coupling_tau, coupling_lambda, run_name=None):
    tau = format_coupling_value(coupling_tau)
    lam = format_coupling_value(coupling_lambda)
    run_prefix = _run_prefix(run_name)
    return f"scout_distance_history_{run_prefix}{tau}_{lam}.json"


def fully_connected_adjacency(num_tasks, device):
    if num_tasks < 2:
        raise ValueError("SCouT needs at least two tasks.")
    adjacency = torch.ones(num_tasks, num_tasks, device=device)
    adjacency.fill_diagonal_(0.0)
    return adjacency


def uniform_task_weights(num_tasks, device):
    if num_tasks < 2:
        raise ValueError("SCouT needs at least two tasks.")
    return torch.full((num_tasks,), 1.0 / num_tasks, device=device)


def _prepare_task_weights(task_weights, num_tasks, device):
    if task_weights is None:
        return uniform_task_weights(num_tasks, device)

    task_weights = torch.as_tensor(task_weights, device=device, dtype=torch.float32)
    if task_weights.shape != (num_tasks,):
        raise ValueError(
            f"Expected task weights with shape {(num_tasks,)}, "
            f"got {tuple(task_weights.shape)}."
        )
    return task_weights


def _trainable_encoder_parameters(model):
    return {
        name: param
        for name, param in model.image_encoder.named_parameters()
        if param.requires_grad
    }


def _matching_trainable_encoder_parameters(models):
    named_params = [_trainable_encoder_parameters(model) for model in models]
    param_names = list(named_params[0].keys())
    for params in named_params[1:]:
        if list(params.keys()) != param_names:
            raise ValueError("All SCouT models must have matching parameters.")
    return named_params, param_names


def scout_loss(models, pretrained_state_dict, coupling_lambda, task_weights=None):
    """Differentiable lambda/2 * sum_k ||w_0 + sum_l a_l u_l - w_k||^2 penalty."""
    num_tasks = len(models)
    device = next(models[0].parameters()).device
    task_weights = _prepare_task_weights(task_weights, num_tasks, device)
    named_params, param_names = _matching_trainable_encoder_parameters(models)

    loss = torch.zeros((), device=device)
    for name in param_names:
        if name not in pretrained_state_dict:
            raise ValueError(f"Pretrained state dict is missing parameter {name}.")
        pretrained_param = pretrained_state_dict[name].to(device=device)
        task_params = [params[name] for params in named_params]
        merged_param = pretrained_param
        for weight, task_param in zip(task_weights, task_params):
            merged_param = merged_param + weight * (task_param - pretrained_param)
        for task_param in task_params:
            loss = loss + (task_param - merged_param).pow(2).sum()

    return 0.5 * coupling_lambda * loss


@torch.no_grad()
def compute_scout_distance(models, task_weights=None):
    """Return the average distance from each task encoder to the merged center.

    This is a diagnostic for the SCouT penalty: for each trainable encoder
    parameter, it computes the weighted task center once and then averages
    ||theta_k - theta_bar||^2 over tasks. If task_weights is not provided, every
    task receives weight 1 / K.
    """
    num_tasks = len(models)
    device = next(models[0].parameters()).device
    task_weights = _prepare_task_weights(task_weights, num_tasks, device)
    named_params, param_names = _matching_trainable_encoder_parameters(models)

    distance = torch.zeros((), device=device)
    for name in param_names:
        task_params = [params[name].detach() for params in named_params]
        center_param = torch.zeros_like(task_params[0])
        for weight, task_param in zip(task_weights, task_params):
            center_param.add_(task_param, alpha=float(weight.item()))
        for task_param in task_params:
            distance = distance + (task_param - center_param).pow(2).sum()

    return (distance / num_tasks).item()
