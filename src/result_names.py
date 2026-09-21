def _run_suffix(run_name):
    return "" if not run_name else f"_{run_name}"


def finetuned_checkpoint_name(finetuning_mode, run_name=None):
    if finetuning_mode == "standard":
        stem = "finetuned"
    elif finetuning_mode == "linear":
        stem = "linear_finetuned"
    elif finetuning_mode == "saft":
        stem = "saft_finetuned"
    elif finetuning_mode == "attention":
        stem = "attention_finetuned"
    elif finetuning_mode == "mergopt":
        stem = "mergopt_finetuned"
    else:
        raise ValueError(f"Unsupported fine-tuning mode: {finetuning_mode}")
    return f"{stem}{_run_suffix(run_name)}.pt"


def training_checkpoint_name(finetuning_mode, step, run_name=None):
    if finetuning_mode == "standard":
        stem = "checkpoint"
    elif finetuning_mode == "linear":
        stem = "linear_checkpoint"
    elif finetuning_mode == "saft":
        stem = "saft_checkpoint"
    elif finetuning_mode == "attention":
        stem = "attention_checkpoint"
    elif finetuning_mode == "mergopt":
        stem = "mergopt_checkpoint"
    else:
        raise ValueError(f"Unsupported fine-tuning mode: {finetuning_mode}")
    return f"{stem}{_run_suffix(run_name)}_{step}.pt"


def single_task_accuracy_name(finetuning_mode, run_name=None):
    if finetuning_mode == "standard":
        return f"ft_accuracies{_run_suffix(run_name)}.json"
    if finetuning_mode == "linear":
        return f"linear_ft_accuracies{_run_suffix(run_name)}.json"
    if finetuning_mode == "saft":
        return f"saft_ft_accuracies{_run_suffix(run_name)}.json"
    if finetuning_mode == "attention":
        return f"attention_ft_accuracies{_run_suffix(run_name)}.json"
    if finetuning_mode == "mergopt":
        return f"mergopt_ft_accuracies{_run_suffix(run_name)}.json"
    if finetuning_mode == "posthoc":
        return f"posthoc_ft_accuracies{_run_suffix(run_name)}.json"
    raise ValueError(f"Unsupported finetuning mode: {finetuning_mode}")


def task_addition_name(finetuning_mode, run_name=None):
    if finetuning_mode == "standard":
        return f"additions{_run_suffix(run_name)}.json"
    if finetuning_mode == "linear":
        return f"linear_additions{_run_suffix(run_name)}.json"
    if finetuning_mode == "saft":
        return f"saft_additions{_run_suffix(run_name)}.json"
    if finetuning_mode == "attention":
        return f"attention_additions{_run_suffix(run_name)}.json"
    if finetuning_mode == "mergopt":
        return f"mergopt_additions{_run_suffix(run_name)}.json"
    if finetuning_mode == "posthoc":
        return f"posthoc_additions{_run_suffix(run_name)}.json"
    raise ValueError(f"Unsupported finetuning mode: {finetuning_mode}")
