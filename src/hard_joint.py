import os


def _run_suffix(run_name):
    return "" if not run_name else f"_{run_name}"


def hard_joint_checkpoint_name(run_name=None):
    return f"hj_finetuned{_run_suffix(run_name)}.pt"


def hard_joint_checkpoint_path(save_dir, run_name=None):
    return os.path.join(save_dir, hard_joint_checkpoint_name(run_name=run_name))


def hard_joint_accuracy_name(run_name=None):
    return f"hj_ft_accuracies{_run_suffix(run_name)}.json"
