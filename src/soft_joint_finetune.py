"""SCouT training implementation for softly coupled vision specialists."""

import copy
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch

from src import utils
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.distributed import (
    cleanup_ddp,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    setup_torchrun,
)
from src.heads import get_classification_head
from src.modeling import ImageClassifier, ImageEncoder
from src.result_names import finetuned_checkpoint_name
from src.soft_joint import (
    soft_joint_checkpoint_name,
    soft_joint_distance_name,
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


def _set_seed(seed):
    if seed is None:
        return
    seed += get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TaskState:
    train_dataset: str
    model: ImageClassifier
    params: list
    data_loader: object
    data_iter: object
    num_batches: int
    epoch: int = 0
    batch_idx: int = 0


def _main_finetuning_mode(args):
    if args.finetuning_mode in [None, "soft_joint"]:
        return "standard"
    return args.finetuning_mode


def _build_image_encoder(args):
    load_path = args.load
    if isinstance(load_path, list):
        load_path = load_path[0] if len(load_path) == 1 else None
    if load_path is not None and load_path.endswith("pt"):
        return utils.torch_load(load_path)
    print("Building image encoder.")
    return ImageEncoder(args)


def _checkpoint_dir(args, train_dataset):
    return os.path.join(args.save, train_dataset)


def _zeroshot_path(args, train_dataset):
    return os.path.join(_checkpoint_dir(args, train_dataset), "zeroshot.pt")


def _finetuned_path(args, train_dataset):
    return os.path.join(
        _checkpoint_dir(args, train_dataset),
        soft_joint_checkpoint_name(
            args.coupling_tau,
            args.coupling_lambda,
            run_name=args.run_name,
        ),
    )


def _independent_init_path(args, train_dataset):
    run_name = getattr(args, "init_from_independent_run", None)
    if not run_name:
        return None
    return os.path.join(
        _checkpoint_dir(args, train_dataset),
        finetuned_checkpoint_name("standard", run_name),
    )


def _ensure_zeroshot_checkpoint(args, train_dataset):
    path = _zeroshot_path(args, train_dataset)
    if os.path.exists(path):
        return
    os.makedirs(_checkpoint_dir(args, train_dataset), exist_ok=True)
    image_encoder = _build_image_encoder(args)
    image_encoder.save(path)


def _distance_history_path(args):
    return os.path.join(
        args.save,
        soft_joint_distance_name(
            args.coupling_tau,
            args.coupling_lambda,
            run_name=args.run_name,
        ),
    )


def _build_task(args, train_dataset):
    task_args = copy.copy(args)
    task_args.train_dataset = train_dataset

    init_path = _independent_init_path(task_args, train_dataset)
    if init_path is None:
        image_encoder = _build_image_encoder(task_args)
    else:
        if not os.path.exists(init_path):
            raise FileNotFoundError(
                f"Could not initialize {train_dataset} from {init_path}."
            )
        print(f"Initializing {train_dataset} from {init_path}.")
        image_encoder = utils.torch_load(init_path)
    classification_head = get_classification_head(task_args, train_dataset)
    model = ImageClassifier(image_encoder, classification_head)
    model.freeze_head()
    model = model.to(task_args.device)

    dataset = get_dataset(
        train_dataset,
        model.train_preprocess,
        location=task_args.data_location,
        batch_size=task_args.batch_size,
    )
    data_loader = get_dataloader(
        dataset, is_train=True, args=task_args, image_encoder=None
    )

    params = [p for p in model.parameters() if p.requires_grad]
    return TaskState(
        train_dataset=train_dataset,
        model=model,
        params=params,
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


def _backward_task_loss(state, args, loss_fn):
    loss_value = 0.0
    data_time = 0.0
    batch_time = 0.0

    for _ in range(args.num_grad_accumulation):
        start_time = time.time()
        inputs, labels, microbatch_data_time = _next_microbatch(state, args)
        data_time += microbatch_data_time

        logits = state.model(inputs)
        loss = loss_fn(logits, labels)
        (loss / args.num_grad_accumulation).backward()
        loss_value += loss.item() / args.num_grad_accumulation
        batch_time += time.time() - start_time

    return {
        "loss": loss_value,
        "data_time": data_time,
        "batch_time": batch_time,
    }


def _normalize_train_datasets(train_datasets):
    datasets = train_datasets or DEFAULT_DATASETS
    if isinstance(datasets, str):
        datasets = [datasets]
    return [
        dataset if dataset.endswith("Val") else f"{dataset}Val"
        for dataset in datasets
    ]


def _clip_gradients(states, all_params, args):
    if args.clip_mode == "noclip":
        return
    if args.clip_mode == "global":
        if not is_distributed():
            torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip_norm)
            return

        total_squared_norm = torch.zeros((), device=args.device)
        for param in all_params:
            if param.grad is not None:
                total_squared_norm.add_(param.grad.detach().float().square().sum())
        torch.distributed.all_reduce(total_squared_norm)
        clip_coefficient = args.grad_clip_norm / (
            total_squared_norm.sqrt() + 1e-6
        )
        clip_coefficient.clamp_(max=1.0)
        for param in all_params:
            if param.grad is not None:
                param.grad.mul_(clip_coefficient.to(param.grad.dtype))
        return
    if args.clip_mode == "indept":
        for state in states:
            torch.nn.utils.clip_grad_norm_(state.params, args.grad_clip_norm)
        return
    raise ValueError("soft_joint_finetune.py supports global, indept, or noclip.")


def _trainable_encoder_parameters(model):
    return {
        name: param
        for name, param in model.image_encoder.named_parameters()
        if param.requires_grad
    }


@torch.no_grad()
def _add_sharded_coupling_gradients(
    models,
    total_tasks,
    coupling_lambda,
    compute_loss=True,
    bucket_cap_mb=64,
):
    """Add the exact uniform-center coupling gradient to locally owned models."""
    named_params = [_trainable_encoder_parameters(model) for model in models]
    param_names = list(named_params[0])
    for params in named_params[1:]:
        if list(params) != param_names:
            raise ValueError("All SCouT models must have matching parameters.")

    bucket_cap_bytes = int(bucket_cap_mb * 1024**2)
    if bucket_cap_bytes <= 0:
        raise ValueError("bucket_cap_mb must be positive.")

    buckets = []
    bucket = []
    bucket_bytes = 0
    bucket_dtype = None
    for name in param_names:
        param = named_params[0][name]
        param_bytes = param.numel() * param.element_size()
        if bucket and (
            bucket_bytes + param_bytes > bucket_cap_bytes
            or param.dtype != bucket_dtype
        ):
            buckets.append(bucket)
            bucket = []
            bucket_bytes = 0
        bucket.append(name)
        bucket_bytes += param_bytes
        bucket_dtype = param.dtype
    if bucket:
        buckets.append(bucket)

    device = next(models[0].parameters()).device
    coupling_loss = torch.zeros((), device=device) if compute_loss else None
    for bucket_names in buckets:
        first_param = named_params[0][bucket_names[0]]
        center = torch.zeros(
            sum(named_params[0][name].numel() for name in bucket_names),
            device=first_param.device,
            dtype=first_param.dtype,
        )
        for params in named_params:
            offset = 0
            for name in bucket_names:
                param = params[name]
                next_offset = offset + param.numel()
                center[offset:next_offset].add_(param.detach().reshape(-1))
                offset = next_offset
        if is_distributed():
            torch.distributed.all_reduce(center)
        center.div_(total_tasks)

        offset = 0
        for name in bucket_names:
            reference_param = named_params[0][name]
            next_offset = offset + reference_param.numel()
            center_param = center[offset:next_offset].view_as(reference_param)
            for params in named_params:
                param = params[name]
                if compute_loss:
                    difference = param.detach() - center_param
                    coupling_loss.add_(
                        difference.square().sum(), alpha=0.5 * coupling_lambda
                    )
                    if param.grad is None:
                        param.grad = difference.mul(coupling_lambda)
                    else:
                        param.grad.add_(difference, alpha=coupling_lambda)
                elif param.grad is None:
                    param.grad = param.detach().sub(center_param).mul_(coupling_lambda)
                else:
                    param.grad.add_(param, alpha=coupling_lambda)
                    param.grad.add_(center_param, alpha=-coupling_lambda)
            offset = next_offset

    if compute_loss and is_distributed():
        torch.distributed.all_reduce(coupling_loss)
    return coupling_loss.item() if compute_loss else 0.0


def _global_step_metrics(step_infos, coupling_loss_value, device):
    values = torch.tensor(
        [
            sum(info["loss"] for info in step_infos),
            sum(info["data_time"] for info in step_infos),
            sum(info["batch_time"] for info in step_infos),
            len(step_infos),
        ],
        dtype=torch.float64,
        device=device,
    )
    if is_distributed():
        torch.distributed.all_reduce(values)
    task_loss, data_time, batch_time, task_count = values.tolist()
    return {
        "task_loss": task_loss,
        "coupling_loss": coupling_loss_value,
        "avg_data_time": data_time / task_count,
        "avg_batch_time": batch_time / task_count,
    }


def soft_joint_finetune(args, train_datasets=None):
    assert args.save is not None, "Please provide a checkpoint directory with --save."
    assert args.num_steps > 0, "--num-steps must be positive."
    assert args.coupling_tau > 0, "--coupling-tau must be positive."
    assert args.coupling_lambda >= 0, "--coupling-lambda must be non-negative."
    assert args.run_name, "--run-name is required for SCouT fine-tuning."
    _set_seed(args.seed)

    main_mode = _main_finetuning_mode(args)
    assert main_mode == "standard", (
        "SCouT uses AdamW over the specialist parameters."
    )

    train_datasets = _normalize_train_datasets(train_datasets)
    assert len(train_datasets) > 1, "SCouT needs at least two tasks."
    world_size = get_world_size()
    rank = get_rank()
    if world_size > len(train_datasets):
        raise ValueError("SCouT needs at least one task per distributed process.")
    local_train_datasets = train_datasets[rank::world_size]

    final_paths = [_finetuned_path(args, dataset) for dataset in train_datasets]
    zeroshot_paths = [_zeroshot_path(args, dataset) for dataset in train_datasets]

    if is_main_process():
        if is_distributed():
            print(
                f"Using task-sharded SCouT across {world_size} GPUs."
            )
        else:
            print("Using SCouT with one global AdamW optimizer.")
        print(
            f"Optimizing sum_k J_k(w_k) + lambda/2 * sum_k ||w_m - w_k||^2 "
            f"with lambda={args.coupling_lambda} lr={args.lr}"
        )
        print(f"Running one coupling update every {args.coupling_tau} main update(s).")
        print(f"Gradient clipping mode: {args.clip_mode}.")
    if is_distributed():
        print(f"Rank {rank} owns: {', '.join(local_train_datasets)}", flush=True)

    loss_fn = LabelSmoothing(args.ls) if args.ls > 0 else torch.nn.CrossEntropyLoss()
    states = [
        _build_task(args, train_dataset)
        for train_dataset in local_train_datasets
    ]
    all_params = [param for state in states for param in state.params]
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.num_steps)

    for state in states:
        os.makedirs(_checkpoint_dir(args, state.train_dataset), exist_ok=True)
        if args.init_from_independent_run:
            _ensure_zeroshot_checkpoint(args, state.train_dataset)
        else:
            state.model.image_encoder.save(_zeroshot_path(args, state.train_dataset))
        state.model.train()

    optimizer.zero_grad(set_to_none=True)

    print_every = 100
    for step in range(args.num_steps):
        current_step = step + 1
        should_log = current_step == 1 or current_step % print_every == 0

        # Backprop each task loss into only that task encoder. We do this
        # task-by-task to avoid keeping all task computation graphs alive.
        step_infos = [_backward_task_loss(state, args, loss_fn) for state in states]

        coupling_loss_value = 0.0
        if current_step % args.coupling_tau == 0:
            coupling_loss_value = _add_sharded_coupling_gradients(
                [state.model for state in states],
                len(train_datasets),
                args.coupling_lambda,
                compute_loss=should_log,
            )

        # AdamW is parameter-wise, so one optimizer per task shard produces the
        # same updates as one optimizer over every specialist.
        scheduler(step)
        _clip_gradients(states, all_params, args)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if should_log:
            metrics = _global_step_metrics(
                step_infos, coupling_loss_value, args.device
            )
            percent_complete = 100 * current_step / args.num_steps
            first_state = states[0]
            if is_main_process():
                print(
                    f"Train Step: {current_step}/{args.num_steps} [{percent_complete:.0f}%]\t"
                    f"Epoch: {first_state.epoch}\t"
                    f"Batch: {first_state.batch_idx}/{first_state.num_batches}\t"
                    f"Task Loss: {metrics['task_loss']:.6f}\t"
                    f"Soft Joint Loss: {metrics['coupling_loss']:.6f}\t"
                    # f"Distance: {distance:.6f}\t"
                    f"Data (t) {metrics['avg_data_time']:.3f}\t"
                    f"Batch (t) {metrics['avg_batch_time']:.3f}",
                    flush=True,
                )

    # save the final model
    for state in states:
        state.model.image_encoder.save(_finetuned_path(args, state.train_dataset))
    if is_distributed():
        torch.distributed.barrier()

    return zeroshot_paths, final_paths


def main():
    args = parse_arguments()
    if args.finetuning_mode is None:
        args.finetuning_mode = "soft_joint"

    if args.save is None:
            args.save = f"checkpoints/{args.model}"

    distributed_started = setup_torchrun()
    if distributed_started:
        args.device = f"cuda:{int(os.environ['LOCAL_RANK'])}"

    try:
        train_datasets = _normalize_train_datasets(args.train_dataset)
        if is_main_process():
            print("=" * 100)
            print(
                f"SCouT fine-tuning {args.model} on {len(train_datasets)} tasks "
                f"for {args.num_steps} steps"
            )
            print("=" * 100)
        soft_joint_finetune(args, train_datasets)
    finally:
        if distributed_started:
            cleanup_ddp()


if __name__ == "__main__":
    main()
