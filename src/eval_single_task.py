import json

from src import utils
from src.args import parse_arguments
from src.eval import eval_single_dataset
from src.hard_mtl import hard_mtl_accuracy_name, hard_mtl_checkpoint_path
from src.linearize import LinearizedImageEncoder
from src.pcgrad import pcgrad_accuracy_name, pcgrad_checkpoint_path
from src.result_names import finetuned_checkpoint_name, single_task_accuracy_name
from src.scout import scout_accuracy_name, scout_checkpoint_name
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.uncertainty_weighting import uw_accuracy_name, uw_checkpoint_path

# This script evaluates single-task performance before merging.
# Example: python -m src.eval_single_task --finetuning-mode=ft_attention --model=ViT-B-32
args = parse_arguments()
if args.finetuning_mode is None:
    raise ValueError(
        "Please specify --finetuning-mode. "
        "Use scout for SCouT checkpoints, hard_mtl/hard_mtl_uw/hard_mtl_pcgrad "
        "for hard-sharing, or independent_ft/ftts/posthoc_ftts/none/saft/"
        "ft_attention/mergopt."
    )

if args.save is None:
    if args.seed is not None:
        args.save = f"checkpoints_{args.seed}/{args.model}"
    else:
        args.save = f"checkpoints/{args.model}"

accuracies = {}


print("*" * 100)
if args.finetuning_mode == "none":
    print("Evaluating pretrained models.")
elif args.finetuning_mode == "independent_ft":
    print("Evaluating Independent FT models.")
elif args.finetuning_mode == "ftts":
    print("Evaluating FTTS models.")
elif args.finetuning_mode == "posthoc_ftts":
    print("Evaluating post-hoc FTTS models.")
elif args.finetuning_mode == "saft":
    print("Evaluating SAFT models.")
elif args.finetuning_mode == "ft_attention":
    print("Evaluating FT-Attention models.")
elif args.finetuning_mode == "mergopt":
    print("Evaluating MergOPT FT models.")
elif args.finetuning_mode == "scout":
    print("Evaluating SCouT models.")
elif args.finetuning_mode == "hard_mtl":
    print("Evaluating Hard MTL model.")
elif args.finetuning_mode == "hard_mtl_uw":
    print("Evaluating Hard MTL + UW model.")
elif args.finetuning_mode == "hard_mtl_pcgrad":
    print("Evaluating Hard MTL + PCGrad model.")

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

shared_image_encoder = None
if args.finetuning_mode in {"hard_mtl", "hard_mtl_uw", "hard_mtl_pcgrad"}:
    checkpoint_path = (
        uw_checkpoint_path(args.save, eval_run_name)
        if args.finetuning_mode == "hard_mtl_uw"
        else (
            pcgrad_checkpoint_path(args.save, eval_run_name)
            if args.finetuning_mode == "hard_mtl_pcgrad"
            else hard_mtl_checkpoint_path(args.save, eval_run_name)
        )
    )
    shared_image_encoder = utils.torch_load(
        checkpoint_path,
        device=args.device,
    )

for dataset in eval_datasets:
    print("*" * 100)
    print(f"Evaluating on {dataset}")

    if args.finetuning_mode in {"hard_mtl", "hard_mtl_uw", "hard_mtl_pcgrad"}:
        image_encoder = shared_image_encoder
    else:
        pretrained_checkpoint = (
            f"{args.save}/{dataset}Val/ftts_zeroshot.pt"
            if args.finetuning_mode == "ftts"
            else f"{args.save}/{dataset}Val/zeroshot.pt"
        )

        if args.finetuning_mode == "scout":
            checkpoint_name = scout_checkpoint_name(
                args.coupling_tau,
                args.coupling_lambda,
                run_name=eval_run_name,
            )
        else:
            checkpoint_mode = (
                args.finetuning_mode
                if args.finetuning_mode
                in {"independent_ft", "ftts", "saft", "ft_attention", "mergopt"}
                else "independent_ft"
            )
            checkpoint_run_name = (
                eval_run_name
                if args.finetuning_mode
                in {"independent_ft", "ftts", "saft", "ft_attention", "mergopt", "posthoc_ftts"}
                else None
            )
            checkpoint_name = finetuned_checkpoint_name(
                checkpoint_mode,
                checkpoint_run_name,
            )
        finetuned_checkpoint = f"{args.save}/{dataset}Val/{checkpoint_name}"

        try:
            task_vector = (
                LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
                if args.finetuning_mode == "ftts"
                else NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            )
        except FileNotFoundError:
            print(f"Error: Could not find {finetuned_checkpoint}.")
            continue

        if args.finetuning_mode == "none":
            image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
        elif args.finetuning_mode in [
            "independent_ft",
            "ftts",
            "saft",
            "ft_attention",
            "mergopt",
            "scout",
        ]:
            image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
        elif args.finetuning_mode == "posthoc_ftts":
            zs_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
            ft_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
            image_encoder = LinearizedImageEncoder(
                init_encoder=zs_encoder, image_encoder=ft_encoder, args=args
            )

    for split in ["test", "val"]:
        # Evaluate
        print("=" * 100)
        print(f"Evaluating on {split} split.")
        eval_dataset = dataset if split == "test" else f"{dataset}Val"

        metrics = eval_single_dataset(image_encoder, eval_dataset, args)
        accuracies[eval_dataset] = metrics["top1"]
        accuracies[f"{eval_dataset}:loss"] = metrics["loss"]


# if args.finetuning_mode == "none":
#     # Evaluate zero-shot accuracy on ImageNet
#     for split in ["ImageNetVal", "ImageNet"]:
#         accuracies[split] = eval_single_dataset(image_encoder, split, args)["top1"]

test_accuracies = [
    accuracies[dataset] for dataset in eval_datasets if dataset in accuracies
]
if test_accuracies:
    accuracies["avg_top1"] = sum(test_accuracies) / len(test_accuracies)
test_losses = [
    accuracies[f"{dataset}:loss"]
    for dataset in eval_datasets
    if f"{dataset}:loss" in accuracies
]
if test_losses:
    accuracies["avg_loss"] = sum(test_losses) / len(test_losses)

# Save results
if args.finetuning_mode == "none":
    save_path = f"{args.save}/zeroshot_accuracies.json"
elif args.finetuning_mode == "independent_ft":
    save_path = f"{args.save}/{single_task_accuracy_name('independent_ft', eval_run_name)}"
elif args.finetuning_mode == "ftts":
    save_path = f"{args.save}/{single_task_accuracy_name('ftts', eval_run_name)}"
elif args.finetuning_mode == "posthoc_ftts":
    save_path = f"{args.save}/{single_task_accuracy_name('posthoc_ftts', eval_run_name)}"
elif args.finetuning_mode == "saft":
    save_path = f"{args.save}/{single_task_accuracy_name('saft', eval_run_name)}"
elif args.finetuning_mode == "ft_attention":
    save_path = f"{args.save}/{single_task_accuracy_name('ft_attention', eval_run_name)}"
elif args.finetuning_mode == "mergopt":
    save_path = f"{args.save}/{single_task_accuracy_name('mergopt', eval_run_name)}"
elif args.finetuning_mode == "scout":
    accuracy_name = scout_accuracy_name(
        args.coupling_tau,
        args.coupling_lambda,
        run_name=eval_run_name,
    )
    save_path = f"{args.save}/{accuracy_name}"
elif args.finetuning_mode == "hard_mtl":
    save_path = f"{args.save}/{hard_mtl_accuracy_name(eval_run_name)}"
elif args.finetuning_mode == "hard_mtl_uw":
    save_path = f"{args.save}/{uw_accuracy_name(eval_run_name)}"
elif args.finetuning_mode == "hard_mtl_pcgrad":
    save_path = f"{args.save}/{pcgrad_accuracy_name(eval_run_name)}"

with open(save_path, "w") as f:
    json.dump(accuracies, f)
