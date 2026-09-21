METHOD_STEMS = {
    "independent_ft": "independent_ft",
    "ftts": "ftts",
    "saft": "saft",
    "ft_attention": "ft_attention",
    "mergopt": "mergopt",
}


def _run_suffix(run_name):
    return "" if not run_name else f"_{run_name}"


def _method_stem(finetuning_mode):
    try:
        return METHOD_STEMS[finetuning_mode]
    except KeyError as error:
        raise ValueError(
            f"Unsupported fine-tuning mode: {finetuning_mode}"
        ) from error


def finetuned_checkpoint_name(finetuning_mode, run_name=None):
    stem = _method_stem(finetuning_mode)
    return f"{stem}_finetuned{_run_suffix(run_name)}.pt"


def training_checkpoint_name(finetuning_mode, step, run_name=None):
    stem = _method_stem(finetuning_mode)
    return f"{stem}_checkpoint{_run_suffix(run_name)}_{step}.pt"


def single_task_accuracy_name(finetuning_mode, run_name=None):
    stem = "posthoc_ftts" if finetuning_mode == "posthoc_ftts" else _method_stem(
        finetuning_mode
    )
    return f"{stem}_accuracies{_run_suffix(run_name)}.json"


def merge_result_name(finetuning_mode, run_name=None):
    stem = "posthoc_ftts" if finetuning_mode == "posthoc_ftts" else _method_stem(
        finetuning_mode
    )
    return f"{stem}_merge{_run_suffix(run_name)}.json"
