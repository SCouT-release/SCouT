import torch

ATTENTION_WEIGHT_NAMES = (
    "in_proj_weight",
    "q_proj_weight",
    "k_proj_weight",
    "v_proj_weight",
    "out_proj.weight",
)

ATTENTION_BIAS_NAMES = (
    "in_proj_bias",
    "bias_k",
    "bias_v",
    "out_proj.bias",
)


def _join_module_parameter_name(module_name, parameter_name):
    return parameter_name if module_name == "" else f"{module_name}.{parameter_name}"


def ft_attention_parameter_names(model, include_bias=False):
    """Return attention projection parameter names for Transformer fine-tuning."""
    parameter_names = set()
    target_names = ATTENTION_WEIGHT_NAMES
    if include_bias:
        target_names = target_names + ATTENTION_BIAS_NAMES

    for module_name, module in model.named_modules():
        if not isinstance(module, torch.nn.MultiheadAttention):
            continue
        module_parameters = dict(module.named_parameters())
        for parameter_name in target_names:
            if parameter_name in module_parameters:
                parameter_names.add(
                    _join_module_parameter_name(module_name, parameter_name)
                )

    return parameter_names


def configure_ft_attention(model, include_bias=False, verbose=True):
    """Freeze all parameters except attention projection weights."""
    trainable_names = ft_attention_parameter_names(model, include_bias=include_bias)
    if not trainable_names:
        raise ValueError(
            "No torch.nn.MultiheadAttention parameters found for attention fine-tuning."
        )

    params = []
    trainable_count = 0
    total_count = 0
    for name, param in model.named_parameters():
        should_train = name in trainable_names
        param.requires_grad_(should_train)
        total_count += param.numel()
        if should_train:
            params.append(param)
            trainable_count += param.numel()

    if not params:
        raise ValueError("Attention fine-tuning found no trainable parameters.")

    if verbose:
        print(
            "Attention fine-tuning: "
            f"{len(params)} tensors, {trainable_count:,}/{total_count:,} "
            "parameters trainable."
        )

    return params
