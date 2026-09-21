import copy
import time
from dataclasses import dataclass

import torch

from src import utils
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.hard_joint import hard_joint_checkpoint_path
from src.heads import get_classification_head
from src.modeling import ImageEncoder, MultiHeadImageClassifier
from src.PCGrad import PCGrad, pcgrad_checkpoint_path
from src.uncertainty_weighting import (
    UncertaintyWeighting,
    save_uw_statistics,
    uw_checkpoint_path,
)
from src.utils import LabelSmoothing, cosine_lr

DEFAULT_DATASETS = [
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



@dataclass
class TaskState:
    train_dataset: str
    head_idx: int
    data_loader: object
    data_iter: object
    num_batches: int
    epoch: int = 0
    batch_idx: int = 0


def _build_image_encoder(args):
    load_path = args.load if isinstance(args.load, str) else None
    if load_path is not None and load_path.endswith("pt"):
        return utils.torch_load(load_path)
    print("Building image encoder.")
    return ImageEncoder(args)


def _normalize_train_datasets(train_datasets):
    datasets = train_datasets or DEFAULT_DATASETS
    if isinstance(datasets, str):
        datasets = [datasets]
    return [
        dataset if dataset.endswith("Val") else f"{dataset}Val"
        for dataset in datasets
    ]


def _build_task_state(args, model, train_dataset, head_idx):
    task_args = copy.copy(args)
    task_args.train_dataset = train_dataset

    dataset = get_dataset(
        train_dataset,
        model.train_preprocess,
        location=task_args.data_location,
        batch_size=task_args.batch_size,
    )
    data_loader = get_dataloader(
        dataset, is_train=True, args=task_args, image_encoder=None
    )

    return TaskState(
        train_dataset=train_dataset,
        head_idx=head_idx,
        data_loader=data_loader,
        data_iter=iter(data_loader),
        num_batches=len(dataset.train_loader),
    )


def _next_microbatch(state, args):
    start_time = time.time()
    try:
        batch = next(state.data_iter)
    except StopIteration:
        state.epoch += 1
        state.batch_idx = 0
        state.data_iter = iter(state.data_loader)
        batch = next(state.data_iter)

    batch = maybe_dictionarize(batch)
    inputs = batch["images"].to(args.device)
    labels = batch["labels"].to(args.device)
    state.batch_idx += 1
    return inputs, labels, time.time() - start_time


def _clip_gradients(params, args):
    if args.clip_mode == "noclip":
        return
    if args.clip_mode != "indept":
        raise ValueError(
            "hard_joint_finetune.py supports --clip-mode indept or noclip."
        )
    torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm)


def _train_one_step(
    model,
    states,
    args,
    loss_fn,
    optimizer,
    scheduler,
    params,
    step,
    uncertainty_weighting=None,
):
    step_losses = []
    step_objectives = []
    data_time = 0.0
    batch_time = 0.0
    loss_scale = len(states) * args.num_grad_accumulation

    for state in states:
        for _ in range(args.num_grad_accumulation):
            start_time = time.time()
            inputs, labels, microbatch_data_time = _next_microbatch(state, args)
            data_time += microbatch_data_time

            logits = model(inputs, state.head_idx)
            loss = loss_fn(logits, labels)
            objective = (
                loss
                if uncertainty_weighting is None
                else uncertainty_weighting(loss, state.head_idx)
            )
            step_losses.append(loss.detach())
            step_objectives.append(objective.detach())
            (objective / loss_scale).backward()
            batch_time += time.time() - start_time

    scheduler(step)
    _clip_gradients(params, args)
    optimizer.step()
    optimizer.zero_grad()

    return {
        "loss": torch.stack(step_losses).mean().item(),
        "objective": torch.stack(step_objectives).mean().item(),
        "data_time": data_time / loss_scale,
        "batch_time": batch_time / loss_scale,
    }


def _train_one_pcgrad_step(
    model, states, args, loss_fn, optimizer, scheduler, params, step, pcgrad
):
    step_losses = []
    task_gradients = []
    data_time = 0.0
    batch_time = 0.0
    num_microbatches = len(states) * args.num_grad_accumulation

    for state in states:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.num_grad_accumulation):
            start_time = time.time()
            inputs, labels, microbatch_data_time = _next_microbatch(state, args)
            data_time += microbatch_data_time

            loss = loss_fn(model(inputs, state.head_idx), labels)
            step_losses.append(loss.detach())
            (loss / args.num_grad_accumulation).backward()
            batch_time += time.time() - start_time

        task_gradients.append(pcgrad.capture())

    pcgrad_info = pcgrad.project_and_assign(task_gradients)
    del task_gradients
    scheduler(step)
    _clip_gradients(params, args)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return {
        "loss": torch.stack(step_losses).mean().item(),
        "objective": torch.stack(step_losses).mean().item(),
        "data_time": data_time / num_microbatches,
        "batch_time": batch_time / num_microbatches,
        **pcgrad_info,
    }


def hard_joint_finetune(args, train_datasets=None):
    assert args.save is not None, "Please provide a checkpoint directory with --save."
    assert args.num_steps > 0, "--num-steps must be positive."
    train_datasets = _normalize_train_datasets(train_datasets)
    assert len(train_datasets) > 1, "Hard-joint fine-tuning needs at least two tasks."
    finetuning_mode = getattr(args, "finetuning_mode", None) or "hard_joint"
    if finetuning_mode not in {"hard_joint", "uw", "pcgrad"}:
        raise ValueError("Expected --finetuning-mode hard_joint, uw, or pcgrad.")
    args.finetuning_mode = finetuning_mode

    if finetuning_mode == "uw":
        print("Using hard-joint multitask fine-tuning with uncertainty weighting.")
    elif finetuning_mode == "pcgrad":
        print("Using hard-joint multitask fine-tuning with PCGrad.")
    else:
        print("Using hard-joint multitask fine-tuning.")
    print(
        f"Training one shared encoder on {len(train_datasets)} tasks "
        f"for {args.num_steps} steps."
    )
    print(f"Gradient clipping mode: {args.clip_mode}.")

    image_encoder = _build_image_encoder(args)
    classification_heads = [
        get_classification_head(args, train_dataset)
        for train_dataset in train_datasets
    ]
    model = MultiHeadImageClassifier(image_encoder, classification_heads)
    model.freeze_head()
    model = model.to(args.device)

    states = [
        _build_task_state(args, model, train_dataset, head_idx)
        for head_idx, train_dataset in enumerate(train_datasets)
    ]

    loss_fn = LabelSmoothing(args.ls) if args.ls > 0 else torch.nn.CrossEntropyLoss()
    params = [p for p in model.image_encoder.parameters() if p.requires_grad]
    uncertainty_weighting = None
    optimizer_params = params
    optimizer_lrs = args.lr
    if finetuning_mode == "uw":
        uncertainty_weighting = UncertaintyWeighting(len(states)).to(args.device)
        optimizer_params = [
            {"params": params, "lr": args.lr, "weight_decay": args.wd},
            {
                "params": uncertainty_weighting.parameters(),
                "lr": args.uw_lr,
                "weight_decay": 0.0,
            },
        ]
        optimizer_lrs = [args.lr, args.uw_lr]
        print(f"UW learning rate: {args.uw_lr:g}.")
    optimizer = torch.optim.AdamW(
        optimizer_params, lr=args.lr, weight_decay=args.wd
    )
    scheduler = cosine_lr(
        optimizer, optimizer_lrs, args.warmup_length, args.num_steps
    )

    model.train()
    optimizer.zero_grad()
    pcgrad = PCGrad(params) if finetuning_mode == "pcgrad" else None

    print_every = 100
    for step in range(args.num_steps):
        if pcgrad is not None:
            step_info = _train_one_pcgrad_step(
                model,
                states,
                args,
                loss_fn,
                optimizer,
                scheduler,
                params,
                step,
                pcgrad,
            )
        else:
            step_info = _train_one_step(
                model,
                states,
                args,
                loss_fn,
                optimizer,
                scheduler,
                params,
                step,
                uncertainty_weighting,
            )
        current_step = step + 1

        if current_step == 1 or current_step % print_every == 0:
            percent_complete = 100 * current_step / args.num_steps
            first_state = states[0]
            message = (
                f"Train Step: {current_step}/{args.num_steps} [{percent_complete:.0f}%]\t"
                f"Epoch: {first_state.epoch}\t"
                f"Batch: {first_state.batch_idx}/{first_state.num_batches}\t"
                f"Avg Loss: {step_info['loss']:.6f}\t"
                f"Data (t) {step_info['data_time']:.3f}\t"
                f"Batch (t) {step_info['batch_time']:.3f}"
            )
            if uncertainty_weighting is not None:
                message += f"\tUW Loss: {step_info['objective']:.6f}"
            if pcgrad is not None:
                message += (
                    f"\tConflict Rate: {step_info['conflict_rate']:.3f}"
                    f"\tGrad Norm: {step_info['gradient_norm']:.3f}"
                )
            print(message, flush=True)

            if uncertainty_weighting is not None:
                stats = uncertainty_weighting.statistics(train_datasets)
                weights = ", ".join(
                    f"{task['dataset']}={task['normalized_weight']:.3f}"
                    for task in stats["tasks"]
                )
                print(f"Normalized UW weights: {weights}", flush=True)

    # for state in states:
    #     eval_single_dataset(model.image_encoder, state.train_dataset, args)

    save_path = (
        uw_checkpoint_path(args.save, args.run_name)
        if uncertainty_weighting is not None
        else (
            pcgrad_checkpoint_path(args.save, args.run_name)
            if pcgrad is not None
            else hard_joint_checkpoint_path(args.save, args.run_name)
        )
    )
    model.image_encoder.save(save_path)
    if uncertainty_weighting is not None:
        save_uw_statistics(
            uncertainty_weighting, train_datasets, args.save, args.run_name
        )
    return save_path


if __name__ == "__main__":
    args = parse_arguments()
    if args.finetuning_mode is None:
        args.finetuning_mode = "hard_joint"

    assert args.finetuning_mode in {"hard_joint", "uw", "pcgrad"}, (
        "hard_joint_finetune.py expects --finetuning-mode=hard_joint, uw, or pcgrad."
    )

    if args.save is None:
        if args.seed is not None:
            args.save = f"checkpoints_{args.seed}/{args.model}"
        else:
            args.save = f"checkpoints/{args.model}"

    train_datasets = _normalize_train_datasets(args.train_dataset)
    print("=" * 100)
    print(
        f"{args.finetuning_mode} fine-tuning {args.model} on {len(train_datasets)} tasks "
        f"for {args.num_steps} steps"
    )
    print("=" * 100)
    hard_joint_finetune(args, train_datasets)
