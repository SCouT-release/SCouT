import json
import os

from src.args import parse_arguments
from src.eval import evaluate_task_vector, evaluate_task_vector_at_coef
from src.fisher import merge_task_vectors as fisher_merge_task_vectors
from src.result_names import (
    finetuned_checkpoint_name,
    single_task_accuracy_name,
    task_addition_name,
)
from src.soft_joint import (
    soft_joint_accuracy_name,
    soft_joint_addition_name,
    soft_joint_checkpoint_name,
)
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.ties import merge_task_vectors as ties_merge_task_vectors
from src.utils import find_optimal_coef
from src.wudi import merge_task_vectors as wudi_merge_task_vectors

# This script evaluate multitask performance after merging
# run this by:
# python -m src.eval_task_addition
#   --finetuning-mode=standard/linear/saft/attention/mergopt/soft_joint --model=ViT-B-32

args = parse_arguments()


def _merge_result_name(filename, merge_mode):
    if merge_mode == "ta":
        return filename
    root, ext = os.path.splitext(filename)
    return f"{root}_{merge_mode}{ext}"


def _merge_task_vectors(task_vectors, pretrained_checkpoint, eval_datasets, args):
    if args.merge_mode == "ta":
        return sum(task_vectors)
    if args.merge_mode == "average":
        return sum(task_vectors) * (1.0 / len(task_vectors))
    if args.merge_mode == "fisher":
        return fisher_merge_task_vectors(
            task_vectors,
            pretrained_checkpoint,
            eval_datasets,
            args,
        )
    if args.merge_mode == "ties":
        return ties_merge_task_vectors(task_vectors, args)
    if args.merge_mode == "wudi":
        return wudi_merge_task_vectors(task_vectors, args)
    raise ValueError(f"Invalid merge mode: {args.merge_mode}")


def _evaluate_merge(task_vector, pretrained_checkpoint, args):
    posthoc_linearization = args.finetuning_mode == "posthoc"
    search_coefficient = args.merge_mode in {"ta", "fisher", "ties"} or (
        args.merge_mode == "wudi" and args.finetuning_mode == "soft_joint"
    )
    if search_coefficient:
        val_metrics = evaluate_task_vector(
            task_vector,
            pretrained_checkpoint,
            args,
            posthoc_linearization=posthoc_linearization,
        )
        optimal_coef = find_optimal_coef(
            val_metrics,
            metric="avg_normalized_top1",
            minimize=False,
        )
    else:
        optimal_coef = 1.0
        val_metrics = {
            optimal_coef: evaluate_task_vector_at_coef(
                task_vector,
                pretrained_checkpoint,
                args,
                optimal_coef,
                posthoc_linearization=posthoc_linearization,
            )
        }
    return val_metrics, float(optimal_coef)


if args.save is None:
    if args.seed is not None:
        args.save = f"checkpoints_{args.seed}/{args.model}"
    else:
        args.save = f"checkpoints/{args.model}"

default_eval_datasets = [
        "CIFAR100",
        "Flowers102",
        "PCAM",
        "FER2013",
        "Cars",           # 7,330
        "DTD",            # 1,692
        # "EuroSAT",        # 16,200
        "GTSRB",            # 35,289
        # "MNIST",            # 55,000
        "RESISC45",       # 17,010
        "SUN397",         # 17,865
        "SVHN",           # 68,257
    ]
eval_datasets = args.eval_datasets or default_eval_datasets
eval_run_name = args.run_name or None


print("*" * 100)
if args.finetuning_mode == "standard":
    print("Evaluating non-linear FT models.")
    ft_accuracies_path = os.path.join(
        args.save,
        single_task_accuracy_name("standard", eval_run_name),
    )
elif args.finetuning_mode == "linear":
    print("Evaluating linear FT models.")
    ft_accuracies_path = os.path.join(
        args.save,
        single_task_accuracy_name("linear", eval_run_name),
    )
elif args.finetuning_mode == "posthoc":
    print("Evaluating post-hoc linearized models.")
    ft_accuracies_path = os.path.join(
        args.save,
        single_task_accuracy_name("posthoc", eval_run_name),
    )
elif args.finetuning_mode == "saft":
    print("Evaluating SAFT models.")
    ft_accuracies_path = os.path.join(
        args.save,
        single_task_accuracy_name("saft", eval_run_name),
    )
elif args.finetuning_mode == "attention":
    print("Evaluating attention-only FT models.")
    ft_accuracies_path = os.path.join(
        args.save,
        single_task_accuracy_name("attention", eval_run_name),
    )
elif args.finetuning_mode == "mergopt":
    print("Evaluating MergOPT FT models.")
    ft_accuracies_path = os.path.join(
        args.save,
        single_task_accuracy_name("mergopt", eval_run_name),
    )
elif args.finetuning_mode == "soft_joint":
    print("Evaluating SCouT models.")
    ft_accuracies_path = os.path.join(
        args.save,
        soft_joint_accuracy_name(
            args.coupling_tau,
            args.coupling_lambda,
            run_name=eval_run_name,
        ),
    )
else:
    raise ValueError(f"Invalid finetuning mode: {args.finetuning_mode}")
print("*" * 100)

with open(ft_accuracies_path) as f:
    args.finetuning_accuracies = json.load(f)

task_vectors = []

for dataset in eval_datasets:
    if args.finetuning_mode == "linear":
        pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_zeroshot.pt"
        checkpoint_name = finetuned_checkpoint_name("linear", eval_run_name)
        finetuned_checkpoint = f"{args.save}/{dataset}Val/{checkpoint_name}"
        task_vectors.append(
            LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        )
    else:
        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        if args.finetuning_mode in {"saft", "attention", "mergopt"}:
            checkpoint_name = finetuned_checkpoint_name(
                args.finetuning_mode,
                eval_run_name,
            )
            finetuned_checkpoint = f"{args.save}/{dataset}Val/{checkpoint_name}"
        elif args.finetuning_mode == "soft_joint":
            checkpoint_name = soft_joint_checkpoint_name(
                args.coupling_tau,
                args.coupling_lambda,
                run_name=eval_run_name,
            )
            finetuned_checkpoint = f"{args.save}/{dataset}Val/{checkpoint_name}"
        else:
            checkpoint_name = finetuned_checkpoint_name("standard", eval_run_name)
            finetuned_checkpoint = f"{args.save}/{dataset}Val/{checkpoint_name}"
        task_vectors.append(
            NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        )

task_vector = _merge_task_vectors(task_vectors, pretrained_checkpoint, eval_datasets, args)
# We use the validation set to choose the optimal coefficient.
args.eval_datasets = [dataset + "Val" for dataset in eval_datasets]
args.control_dataset = None

val_metrics, optimal_coef = _evaluate_merge(task_vector, pretrained_checkpoint, args)

# Evaluate on the test set with the optimal coefficient.
args.eval_datasets = [dataset for dataset in eval_datasets]
test_metrics = evaluate_task_vector_at_coef(
    task_vector,
    pretrained_checkpoint,
    args,
    optimal_coef,
    posthoc_linearization=args.finetuning_mode == "posthoc",
)
test_metrics["best_coef"] = optimal_coef
test_metrics["merge_mode"] = args.merge_mode
test_metrics["pre_merge_avg_top1"] = sum(
    args.finetuning_accuracies[dataset] for dataset in eval_datasets
) / len(eval_datasets)
pre_merge_losses = [
    args.finetuning_accuracies[f"{dataset}:loss"]
    for dataset in eval_datasets
    if f"{dataset}:loss" in args.finetuning_accuracies
]
if pre_merge_losses:
    test_metrics["pre_merge_avg_loss"] = sum(pre_merge_losses) / len(pre_merge_losses)

print("=" * 100)
print(f"Test normalized accuracy: {test_metrics['avg_normalized_top1']}")
print(f"Test absolute accuracy: {test_metrics['avg_top1']}")
print(f"Test average loss: {test_metrics['avg_loss']}")
additive_accuracies = {"test": test_metrics, "val": val_metrics}

if args.finetuning_mode == "standard":
    save_name = task_addition_name("standard", eval_run_name)
elif args.finetuning_mode == "linear":
    save_name = task_addition_name("linear", eval_run_name)
elif args.finetuning_mode == "posthoc":
    save_name = task_addition_name("posthoc", eval_run_name)
elif args.finetuning_mode == "saft":
    save_name = task_addition_name("saft", eval_run_name)
elif args.finetuning_mode == "attention":
    save_name = task_addition_name("attention", eval_run_name)
elif args.finetuning_mode == "mergopt":
    save_name = task_addition_name("mergopt", eval_run_name)
elif args.finetuning_mode == "soft_joint":
    save_name = soft_joint_addition_name(
        args.coupling_tau, args.coupling_lambda, run_name=eval_run_name
    )
save_file = f"{args.save}/{_merge_result_name(save_name, args.merge_mode)}"
with open(save_file, "w") as f:
    json.dump(additive_accuracies, f, indent=4)
    print(f"Saved task addition results to {save_file}")
