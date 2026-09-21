import os


def _run_suffix(run_name):
    return "" if not run_name else f"_{run_name}"


def hard_mtl_checkpoint_name(run_name=None):
    return f"hard_mtl_finetuned{_run_suffix(run_name)}.pt"


def hard_mtl_checkpoint_path(save_dir, run_name=None):
    return os.path.join(save_dir, hard_mtl_checkpoint_name(run_name=run_name))


def hard_mtl_accuracy_name(run_name=None):
    return f"hard_mtl_accuracies{_run_suffix(run_name)}.json"
