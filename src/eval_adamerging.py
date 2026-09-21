import json
import math
import os

import torch
import torch.nn.functional as F

from src import utils
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.eval import evaluate_task_vector_at_coef
from src.heads import get_classification_head
from src.scout import (
    scout_accuracy_name,
    scout_adamerging_name,
    scout_checkpoint_name,
)
from src.task_vectors import NonLinearTaskVector

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call


DEFAULT_EVAL_DATASETS = [
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SUN397",
    "SVHN",
]


class CyclingLoader:
    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(loader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def _logit(value):
    eps = 1e-6
    value = min(max(value, eps), 1.0 - eps)
    return math.log(value / (1.0 - value))


def _state_to_cpu(state_dict):
    return {key: value.detach().cpu() for key, value in state_dict.items()}


def _task_vectors_to_cpu(task_vectors):
    return [
        {key: value.detach().cpu() for key, value in task_vector.vector.items()}
        for task_vector in task_vectors
    ]


def _merged_state_dict(base_state, task_vector_states, coefficients, device):
    merged_state = {}
    for key, base_value in base_state.items():
        merged_value = base_value
        if key in task_vector_states[0]:
            for coefficient, task_vector_state in zip(
                coefficients, task_vector_states
            ):
                merged_value = merged_value + coefficient * task_vector_state[key]
        merged_state[key] = merged_value.to(device)
    return merged_state


def _weighted_task_vector(task_vectors, coefficients):
    merged_task_vector = None
    for coefficient, task_vector in zip(coefficients, task_vectors):
        weighted_task_vector = task_vector * float(coefficient)
        if merged_task_vector is None:
            merged_task_vector = weighted_task_vector
        else:
            merged_task_vector = merged_task_vector + weighted_task_vector
    return merged_task_vector


def _load_scout_task_vectors(args, eval_datasets, run_name):
    task_vectors = []
    pretrained_checkpoint = None
    for dataset in eval_datasets:
        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        finetuned_checkpoint = (
            f"{args.save}/{dataset}Val/"
            f"{scout_checkpoint_name(args.coupling_tau, args.coupling_lambda, run_name=run_name)}"
        )
        if not os.path.exists(finetuned_checkpoint):
            raise FileNotFoundError(f"Missing checkpoint: {finetuned_checkpoint}")
        task_vectors.append(
            NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        )
    return pretrained_checkpoint, task_vectors


def _make_validation_loaders(args, image_encoder, eval_datasets):
    batch_size = args.adamerging_batch_size or args.batch_size
    loaders = {}
    for dataset_name in eval_datasets:
        val_dataset_name = dataset_name + "Val"
        dataset = get_dataset(
            val_dataset_name,
            image_encoder.val_preprocess,
            location=args.data_location,
            batch_size=batch_size,
        )
        loaders[dataset_name] = CyclingLoader(
            get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
        )
    return loaders


def _make_classification_heads(args, eval_datasets):
    heads = {}
    num_classes = {}
    for dataset_name in eval_datasets:
        head = get_classification_head(args, dataset_name).to(args.device)
        head.eval()
        head.requires_grad_(False)
        heads[dataset_name] = head
        num_classes[dataset_name] = head.weight.shape[0]
    return heads, num_classes


def _supervised_adamerging(args, pretrained_checkpoint, task_vectors, eval_datasets):
    if not 0.0 < args.adamerging_prior < 1.0:
        raise ValueError("--adamerging-prior must be strictly between 0 and 1.")

    image_encoder = utils.torch_load(pretrained_checkpoint)
    image_encoder.eval()
    image_encoder.to(args.device)
    image_encoder.requires_grad_(False)

    base_state = _state_to_cpu(image_encoder.state_dict())
    task_vector_states = _task_vectors_to_cpu(task_vectors)
    val_loaders = _make_validation_loaders(args, image_encoder, eval_datasets)
    heads, num_classes = _make_classification_heads(args, eval_datasets)

    raw_coefficients = torch.nn.Parameter(
        torch.full(
            (len(eval_datasets),),
            _logit(args.adamerging_prior),
            dtype=torch.float32,
        )
    )
    optimizer = torch.optim.Adam([raw_coefficients], lr=args.adamerging_lr)

    for step in range(args.adamerging_steps):
        coefficients = torch.sigmoid(raw_coefficients)
        merged_state = _merged_state_dict(
            base_state, task_vector_states, coefficients, args.device
        )

        task_losses = []
        for dataset_name in eval_datasets:
            batch = maybe_dictionarize(val_loaders[dataset_name].next())
            images = batch["images"].to(args.device)
            labels = batch["labels"].to(args.device)

            features = functional_call(image_encoder, merged_state, (images,))
            logits = heads[dataset_name](features)
            loss = F.cross_entropy(logits, labels)
            loss = loss / math.log(num_classes[dataset_name])
            task_losses.append(loss)

        balanced_loss = torch.stack(task_losses).mean()
        regularizer = (
            (coefficients.to(args.device) - args.adamerging_prior).pow(2).mean()
        )
        loss = balanced_loss + args.adamerging_reg * regularizer

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        if args.adamerging_log_every > 0 and (
            step == 0 or (step + 1) % args.adamerging_log_every == 0
        ):
            coef_text = ", ".join(f"{c:.4f}" for c in coefficients.detach().tolist())
            print(
                f"AdaMerging step {step + 1}/{args.adamerging_steps}: "
                f"loss={loss_value:.4f}, balanced_ce={float(balanced_loss.detach().cpu().item()):.4f}, "
                f"reg={float(regularizer.detach().cpu().item()):.4f}, coeffs=[{coef_text}]"
            )

    final_coefficients = torch.sigmoid(raw_coefficients).detach().cpu()
    return final_coefficients.numpy().astype(float)


def main():
    args = parse_arguments()
    if args.finetuning_mode != "scout":
        raise ValueError("eval_adamerging.py currently expects --finetuning-mode=scout.")

    if args.save is None:
        if args.seed is not None:
            args.save = f"checkpoints_{args.seed}/{args.model}"
        else:
            args.save = f"checkpoints/{args.model}"

    eval_datasets = args.eval_datasets or DEFAULT_EVAL_DATASETS
    run_name = args.run_name or None

    print("*" * 100)
    print("Evaluating SCouT models with supervised task-wise AdaMerging.")
    print("*" * 100)

    ft_accuracies_path = os.path.join(
        args.save,
        scout_accuracy_name(
            args.coupling_tau,
            args.coupling_lambda,
            run_name=run_name,
        ),
    )
    with open(ft_accuracies_path) as f:
        args.finetuning_accuracies = json.load(f)

    pretrained_checkpoint, task_vectors = _load_scout_task_vectors(
        args, eval_datasets, run_name
    )
    coefficients = _supervised_adamerging(
        args, pretrained_checkpoint, task_vectors, eval_datasets
    )
    coefficient_by_task = {
        dataset_name: float(coefficient)
        for dataset_name, coefficient in zip(eval_datasets, coefficients)
    }
    print("Learned AdaMerging coefficients:")
    print(json.dumps(coefficient_by_task, indent=4))

    task_vector = _weighted_task_vector(task_vectors, coefficients)
    args.control_dataset = None

    args.eval_datasets = [dataset + "Val" for dataset in eval_datasets]
    val_metrics = evaluate_task_vector_at_coef(
        task_vector,
        pretrained_checkpoint,
        args,
        scaling_coef=1.0,
    )

    args.eval_datasets = list(eval_datasets)
    test_metrics = evaluate_task_vector_at_coef(
        task_vector,
        pretrained_checkpoint,
        args,
        scaling_coef=1.0,
    )
    test_metrics["pre_merge_avg_top1"] = sum(
        args.finetuning_accuracies[dataset] for dataset in eval_datasets
    ) / len(eval_datasets)

    val_metrics["adamerging_coefficients"] = coefficient_by_task
    test_metrics["adamerging_coefficients"] = coefficient_by_task

    print("=" * 100)
    print(f"Test normalized accuracy: {test_metrics['avg_normalized_top1']}")
    print(f"Test absolute accuracy: {test_metrics['avg_top1']}")

    results = {
        "test": test_metrics,
        "val": val_metrics,
        "coefficients": coefficient_by_task,
        "adamerging": {
            "steps": args.adamerging_steps,
            "lr": args.adamerging_lr,
            "prior": args.adamerging_prior,
            "reg": args.adamerging_reg,
        },
    }

    save_file = os.path.join(
        args.save,
        scout_adamerging_name(
            args.coupling_tau,
            args.coupling_lambda,
            run_name=run_name,
        ),
    )
    with open(save_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved AdaMerging results to {save_file}")


if __name__ == "__main__":
    main()
