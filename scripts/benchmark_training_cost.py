#!/usr/bin/env python3
"""Benchmark training-step time and peak VRAM for the repository's FT methods.

One benchmark global step consumes one minibatch (or the configured number of
gradient-accumulation minibatches) from every task.  Run directly for one GPU,
or with ``torchrun`` to shard tasks across GPUs::

    python scripts/benchmark_training_cost.py
    torchrun --standalone --nproc-per-node=2 \
        scripts/benchmark_training_cost.py

The requested paper name "FTTS" is used in reports for the repository's
linearized/tangent fine-tuning implementation (``LinearizedImageEncoder``).
The normal training entry points do not enable AMP, so this benchmark also
uses full precision.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import os
import random
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

# Make ``python scripts/benchmark_training_cost.py`` work from any cwd.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.attention_ft import configure_attention_finetuning  # noqa: E402
from src.datasets.common import get_dataloader, maybe_dictionarize  # noqa: E402
from src.datasets.registry import get_dataset  # noqa: E402
from src.distributed import (  # noqa: E402
    cleanup_ddp,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    setup_torchrun,
)
from src.heads import get_classification_head  # noqa: E402
from src.linearize import LinearizedImageEncoder  # noqa: E402
from src.mergopt import (  # noqa: E402
    DEFAULT_MERGOPT_ALPHAS,
    MergOPT,
    parse_mergopt_alphas,
)
from src.modeling import (  # noqa: E402
    ImageClassifier,
    ImageEncoder,
    MultiHeadImageClassifier,
)
from src.sam import SAM  # noqa: E402
from src.soft_joint_finetune import (  # noqa: E402
    DEFAULT_DATASETS,
    _add_sharded_coupling_gradients,
)
from src.utils import LabelSmoothing, cosine_lr  # noqa: E402

METHODS = (
    "independent_ft",
    "ftts",
    "ft_attention",
    "saft",
    "mergopt",
    "socoft",
    "hard_mtl",
)
INDEPENDENT_METHODS = frozenset(METHODS[:5])
METHOD_LABELS = {
    "independent_ft": "Independent FT",
    "ftts": "FTTS",
    "ft_attention": "FT-Attention",
    "saft": "SAFT",
    "mergopt": "MergOPT",
    "socoft": "SCouT",
    "hard_mtl": "Hard MTL",
}
METHOD_ALIASES = {
    "independent": "independent_ft",
    "standard": "independent_ft",
    "tft": "ftts",
    "linear": "ftts",
    "attention": "ft_attention",
    "scout": "socoft",
    "soft_joint": "socoft",
    "hard_joint": "hard_mtl",
}
DEFAULT_ARCHITECTURES = ("ViT-B-32",)


@dataclass
class TaskState:
    name: str
    model: ImageClassifier
    params: list[torch.nn.Parameter]
    data_loader: object
    data_iter: object
    head_idx: int | None = None


@dataclass
class MethodRuntime:
    states: list[TaskState]
    prepare_batches: Callable[[], list[list[tuple[torch.Tensor, torch.Tensor]]]]
    train_step: Callable[[list[list[tuple[torch.Tensor, torch.Tensor]]], int, bool], dict]
    trainable_parameter_count: int


def _comma_separated(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def _parse_methods(value: str) -> list[str]:
    methods = []
    for requested in _comma_separated(value):
        canonical = METHOD_ALIASES.get(requested.lower(), requested.lower())
        if canonical not in METHODS:
            raise argparse.ArgumentTypeError(
                f"unknown method {requested!r}; choose from {', '.join(METHODS)}"
            )
        if canonical not in methods:
            methods.append(canonical)
    return methods


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architectures",
        type=_comma_separated,
        default=list(DEFAULT_ARCHITECTURES),
        help="Comma-separated OpenCLIP architectures (default: ViT-B-32).",
    )
    parser.add_argument(
        "--methods",
        type=_parse_methods,
        default=list(METHODS),
        help="Comma-separated methods; aliases tft/linear, attention, and *_joint work.",
    )
    parser.add_argument(
        "--train-dataset",
        type=_comma_separated,
        default=list(DEFAULT_DATASETS),
        help="Comma-separated task set (the repository defaults are used by default).",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Use only the first N tasks from --train-dataset (default: use all).",
    )
    parser.add_argument("--data-location", default=os.path.expanduser("~/data"))
    parser.add_argument(
        "--openclip-cachedir",
        default=os.path.expanduser("~/openclip-cachedir/open_clip"),
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--head-checkpoint-root",
        default="checkpoints",
        help=(
            "Root containing cached head_<dataset>Val.pt files; "
            "no training checkpoints are written."
        ),
    )
    parser.add_argument("--output-dir", default="results/training_cost_comparison")
    parser.add_argument("--output-prefix", default="training_cost_comparison")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--num-grad-accumulation", type=int, default=1)
    parser.add_argument(
        "--benchmark-warmup-steps",
        "--benchmark_warmup_steps",
        dest="benchmark_warmup_steps",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--benchmark-measure-steps",
        "--benchmark_measure_steps",
        dest="benchmark_measure_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=2000,
        help=(
            "Length of the normal cosine schedule (the benchmark still stops "
            "after warm-up+measured steps)."
        ),
    )
    parser.add_argument(
        "--warmup-length", "--warmup_length", dest="warmup_length", type=int, default=500
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--ls", type=float, default=0.0)
    parser.add_argument("--clip-mode", choices=("noclip", "indept", "global"), default="noclip")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--saft-rho", type=float, default=0.5)
    parser.add_argument("--mergopt-mu", type=float, default=0.0)
    parser.add_argument("--mergopt-b", type=float, default=0.0005)
    parser.add_argument("--mergopt-k-max", type=int, default=None)
    parser.add_argument(
        "--mergopt-alphas",
        type=parse_mergopt_alphas,
        default=DEFAULT_MERGOPT_ALPHAS,
    )
    parser.add_argument("--coupling-tau", type=int, default=1)
    parser.add_argument("--coupling-lambda", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("--batch-size must be positive and --num-workers nonnegative")
    if args.num_grad_accumulation < 1:
        parser.error("--num-grad-accumulation must be positive")
    if args.benchmark_warmup_steps < 1:
        parser.error(
            "--benchmark-warmup-steps must be at least 1 "
            "(the first step runs sanity checks)"
        )
    if args.benchmark_measure_steps < 1:
        parser.error("--benchmark-measure-steps must be positive")
    if args.num_steps < args.benchmark_warmup_steps + args.benchmark_measure_steps:
        parser.error("--num-steps must cover all benchmark warm-up and measured steps")
    if args.warmup_length < 0 or args.warmup_length >= args.num_steps:
        parser.error("--warmup-length must be nonnegative and smaller than --num-steps")
    if args.coupling_tau < 1 or args.coupling_lambda < 0:
        parser.error("SCouT requires positive --coupling-tau and nonnegative --coupling-lambda")
    if args.num_tasks is not None:
        if args.num_tasks < 2:
            parser.error("--num-tasks must be at least 2")
        if args.num_tasks > len(args.train_dataset):
            parser.error("--num-tasks cannot exceed the number of available tasks")
        args.train_dataset = args.train_dataset[: args.num_tasks]
    if len(args.train_dataset) < 2:
        parser.error("the efficiency benchmark requires at least two tasks")
    if args.clip_mode == "global" and any(
        method != "socoft" for method in args.methods
    ):
        parser.error(
            "--clip-mode=global is implemented only for SCouT; use noclip/indept "
            "for a cross-method benchmark"
        )
    return args


def _normalize_datasets(datasets: Iterable[str]) -> list[str]:
    return [name if name.endswith("Val") else f"{name}Val" for name in datasets]


def _set_seed(seed: int, rank_offset: bool = True) -> None:
    effective_seed = seed + (get_rank() if rank_offset else 0)
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)


def _synchronize_workers(device: torch.device) -> None:
    torch.cuda.synchronize(device)
    if is_distributed():
        torch.distributed.barrier()


def _global_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if is_distributed():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return float(tensor.item())


def _global_sum_int(value: int, device: torch.device) -> int:
    tensor = torch.tensor(value, dtype=torch.int64, device=device)
    if is_distributed():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return int(tensor.item())


def _global_all(flag: bool, device: torch.device) -> bool:
    tensor = torch.tensor(int(flag), dtype=torch.int64, device=device)
    if is_distributed():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
    return bool(tensor.item())


def _per_gpu_values(value: float, device: torch.device) -> list[float]:
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    if not is_distributed():
        return [float(tensor.item())]
    gathered = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    torch.distributed.all_gather(gathered, tensor)
    return [float(item.item()) for item in gathered]


def _clip_parameters(params: list[torch.nn.Parameter], args: argparse.Namespace) -> None:
    if args.clip_mode == "noclip":
        return
    # For independent specialists, global and independent clipping coincide for
    # one task. SCouT handles the distinction separately.
    torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm)


def _loss_function(args: argparse.Namespace) -> torch.nn.Module:
    return LabelSmoothing(args.ls) if args.ls > 0 else torch.nn.CrossEntropyLoss()


def _architecture_args(
    args: argparse.Namespace, architecture: str, device: torch.device
) -> argparse.Namespace:
    # Existing model/head/dataset helpers consume this familiar argument shape.
    task_args = argparse.Namespace(**vars(args))
    task_args.model = architecture
    task_args.device = str(device)
    task_args.save = str(Path(args.head_checkpoint_root) / architecture)
    task_args.load = None
    task_args.train_dataset = None
    task_args.warmup_length = args.warmup_length
    return task_args


def _build_loader(task_args: argparse.Namespace, model, dataset_name: str):
    dataset = get_dataset(
        dataset_name,
        model.train_preprocess,
        location=task_args.data_location,
        batch_size=task_args.batch_size,
        num_workers=task_args.num_workers,
    )
    return get_dataloader(dataset, is_train=True, args=task_args, image_encoder=None)


def _build_specialist(
    task_args: argparse.Namespace,
    dataset_name: str,
    mode: str,
) -> TaskState:
    image_encoder = (
        LinearizedImageEncoder(task_args, keep_lang=False)
        if mode == "ftts"
        else ImageEncoder(task_args)
    )
    head = get_classification_head(task_args, dataset_name)
    model = ImageClassifier(image_encoder, head)
    model.freeze_head()
    model = model.to(task_args.device)
    params = (
        configure_attention_finetuning(model, verbose=is_main_process())
        if mode == "ft_attention"
        else [param for param in model.parameters() if param.requires_grad]
    )
    loader = _build_loader(task_args, model, dataset_name)
    model.train()
    return TaskState(dataset_name, model, params, loader, iter(loader))


def _next_cpu_batch(state: TaskState):
    try:
        batch = next(state.data_iter)
    except StopIteration:
        state.data_iter = iter(state.data_loader)
        batch = next(state.data_iter)
    return maybe_dictionarize(batch)


def _shutdown_runtime_data_workers(runtime: MethodRuntime | None) -> None:
    """Stop DataLoader workers before releasing a benchmark runtime.

    A full benchmark creates many loaders in succession.  Relying on Python's
    interpreter-shutdown cleanup leaves multiprocessing queue feeder threads
    racing with closed file descriptors, which can print ``Bad file
    descriptor`` and ``semaphore or lock released too many times`` tracebacks
    after an otherwise successful run.
    """
    if runtime is None:
        return
    for state in runtime.states:
        iterator = state.data_iter
        shutdown_workers = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown_workers):
            shutdown_workers()
        state.data_iter = None


def _prepare_batches(states: list[TaskState], args: argparse.Namespace, device: torch.device):
    prepared = []
    for state in states:
        microbatches = []
        for _ in range(args.num_grad_accumulation):
            batch = _next_cpu_batch(state)
            # Only CPU-side dataloader work is prefetched. Device transfer stays
            # in the training step, as in the repository's normal pipelines.
            microbatches.append((batch["images"], batch["labels"]))
        prepared.append(microbatches)
    return prepared


def _to_device(microbatches, device: torch.device):
    return [
        (inputs.to(device), labels.to(device))
        for inputs, labels in microbatches
    ]


def _backward_microbatches(model, microbatches, loss_fn, accumulation: int):
    losses = []
    for inputs, labels in microbatches:
        loss = loss_fn(model(inputs), labels)
        (loss / accumulation).backward()
        losses.append(loss.detach())
    return torch.stack(losses).mean()


def _has_gradient(params: Iterable[torch.nn.Parameter]) -> bool:
    return any(param.grad is not None for param in params)


def _finite_loss(losses: list[torch.Tensor]) -> bool:
    return bool(torch.stack(losses).isfinite().all().item())


def _build_specialist_runtime(
    method: str,
    task_args: argparse.Namespace,
    local_datasets: list[str],
    total_tasks: int,
    device: torch.device,
) -> MethodRuntime:
    states = [_build_specialist(task_args, name, method) for name in local_datasets]
    loss_fn = _loss_function(task_args)
    optimizers = []
    schedulers = []
    k_max = task_args.mergopt_k_max or total_tasks
    for state in states:
        if method == "saft":
            optimizer = SAM(
                state.params,
                torch.optim.AdamW,
                lr=task_args.lr,
                weight_decay=task_args.wd,
                rho=task_args.saft_rho,
                adaptive=True,
            )
            scheduled_optimizer = optimizer.base_optimizer
        elif method == "mergopt":
            optimizer = MergOPT(
                state.params,
                torch.optim.AdamW,
                lr=task_args.lr,
                weight_decay=task_args.wd,
                mu=task_args.mergopt_mu,
                b=task_args.mergopt_b,
                k_max=k_max,
                alphas=task_args.mergopt_alphas,
            )
            scheduled_optimizer = optimizer
        else:
            optimizer = torch.optim.AdamW(state.params, lr=task_args.lr, weight_decay=task_args.wd)
            scheduled_optimizer = optimizer
        optimizer.zero_grad()
        optimizers.append(optimizer)
        schedulers.append(
            cosine_lr(
                scheduled_optimizer,
                task_args.lr,
                task_args.warmup_length,
                task_args.num_steps,
            )
        )

    def prepare():
        return _prepare_batches(states, task_args, device)

    def train_step(batches, step: int, sanity: bool = False):
        losses = []
        gradients_present = True
        optimizer_steps = 0
        extra_operations = 0
        for state, microbatches, optimizer, scheduler in zip(
            states, batches, optimizers, schedulers
        ):
            device_microbatches = _to_device(microbatches, device)
            if method == "mergopt":
                closure_calls = 0
                closure_has_grad = False

                def closure(
                    state=state,
                    microbatches=device_microbatches,
                    optimizer=optimizer,
                ):
                    nonlocal closure_calls, closure_has_grad
                    closure_calls += 1
                    optimizer.zero_grad()
                    closure_loss = _backward_microbatches(
                        state.model,
                        microbatches,
                        loss_fn,
                        task_args.num_grad_accumulation,
                    )
                    closure_has_grad = _has_gradient(state.params)
                    _clip_parameters(state.params, task_args)
                    return closure_loss

                scheduler(step)
                loss = optimizer.step(closure)
                optimizer_steps += 1
                extra_operations += closure_calls
                gradients_present = gradients_present and closure_has_grad
                losses.append(loss.detach())
                optimizer.zero_grad()
                continue

            loss = _backward_microbatches(
                state.model,
                device_microbatches,
                loss_fn,
                task_args.num_grad_accumulation,
            )
            gradients_present = gradients_present and _has_gradient(state.params)
            losses.append(loss)
            scheduler(step)
            _clip_parameters(state.params, task_args)

            if method == "saft":
                closure_calls = 0

                def closure(state=state, microbatches=device_microbatches):
                    nonlocal closure_calls
                    closure_calls += 1
                    return _backward_microbatches(
                        state.model,
                        microbatches,
                        loss_fn,
                        task_args.num_grad_accumulation,
                    )

                optimizer.step(closure)
                extra_operations += closure_calls
            else:
                optimizer.step()
            optimizer_steps += 1
            optimizer.zero_grad()
            del device_microbatches

        return {
            "finite_loss": _finite_loss(losses) if sanity else True,
            "gradients_present": gradients_present,
            "optimizer_steps": optimizer_steps,
            "extra_operations": extra_operations,
            "task_minibatches": sum(len(task_batches) for task_batches in batches),
        }

    return MethodRuntime(
        states,
        prepare,
        train_step,
        sum(param.numel() for state in states for param in state.params),
    )


def _clip_socoft(
    states: list[TaskState], all_params: list[torch.nn.Parameter], args: argparse.Namespace
) -> None:
    if args.clip_mode == "noclip":
        return
    if args.clip_mode == "indept":
        for state in states:
            torch.nn.utils.clip_grad_norm_(state.params, args.grad_clip_norm)
        return
    if not is_distributed():
        torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip_norm)
        return

    squared_norm = torch.zeros((), device=all_params[0].device)
    for param in all_params:
        if param.grad is not None:
            squared_norm.add_(param.grad.detach().float().square().sum())
    torch.distributed.all_reduce(squared_norm)
    coefficient = (args.grad_clip_norm / (squared_norm.sqrt() + 1e-6)).clamp(max=1.0)
    for param in all_params:
        if param.grad is not None:
            param.grad.mul_(coefficient.to(param.grad.dtype))


def _build_socoft_runtime(
    task_args: argparse.Namespace,
    local_datasets: list[str],
    total_tasks: int,
    device: torch.device,
) -> MethodRuntime:
    states = [_build_specialist(task_args, name, "socoft") for name in local_datasets]
    loss_fn = _loss_function(task_args)
    all_params = [param for state in states for param in state.params]
    optimizer = torch.optim.AdamW(all_params, lr=task_args.lr, weight_decay=task_args.wd)
    scheduler = cosine_lr(
        optimizer, task_args.lr, task_args.warmup_length, task_args.num_steps
    )
    optimizer.zero_grad(set_to_none=True)
    def prepare():
        return _prepare_batches(states, task_args, device)

    def train_step(batches, step: int, sanity: bool = False):
        losses = []
        for state, microbatches in zip(states, batches):
            task_losses = []
            for cpu_inputs, cpu_labels in microbatches:
                inputs = cpu_inputs.to(device)
                labels = cpu_labels.to(device)
                loss = loss_fn(state.model(inputs), labels)
                (loss / task_args.num_grad_accumulation).backward()
                task_losses.append(loss.detach())
                del inputs, labels
            losses.append(torch.stack(task_losses).mean())

        coupling_executed = (step + 1) % task_args.coupling_tau == 0
        if coupling_executed:
            _add_sharded_coupling_gradients(
                [state.model for state in states],
                total_tasks,
                task_args.coupling_lambda,
                compute_loss=False,
            )

        gradients_present = _has_gradient(all_params)
        scheduler(step)
        _clip_socoft(states, all_params, task_args)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return {
            "finite_loss": _finite_loss(losses) if sanity else True,
            "gradients_present": gradients_present,
            "optimizer_steps": 1,
            "extra_operations": int(coupling_executed),
            "task_minibatches": sum(len(task_batches) for task_batches in batches),
        }

    return MethodRuntime(
        states,
        prepare,
        train_step,
        sum(param.numel() for param in all_params),
    )


def _build_hard_mtl_runtime(
    task_args: argparse.Namespace,
    local_datasets: list[str],
    device: torch.device,
    total_tasks: int,
) -> MethodRuntime:
    # Build the encoder identically on every rank before applying rank-specific
    # data seeds. DDP then performs the shared-model gradient reduction.
    _set_seed(task_args.seed, rank_offset=False)
    image_encoder = ImageEncoder(task_args)

    # ``ImageEncoder`` removes OpenCLIP's text transformer, but OpenCLIP keeps
    # several other text-only parameters (for example, token/positional
    # embeddings, text_projection, and logit_scale).  ``encode_image`` never
    # uses them.  Leaving them trainable makes DDP expect gradients that can
    # never be produced and causes the next microbatch to fail with
    # "Expected to have finished reduction in the prior iteration".
    visual_parameter_ids = {
        id(parameter) for parameter in image_encoder.model.visual.parameters()
    }
    for parameter in image_encoder.parameters():
        if id(parameter) not in visual_parameter_ids:
            parameter.requires_grad_(False)

    heads = [get_classification_head(task_args, name) for name in local_datasets]
    model = MultiHeadImageClassifier(image_encoder, heads)
    model.freeze_head()
    model = model.to(device)
    if is_distributed():
        model.image_encoder = torch.nn.parallel.DistributedDataParallel(
            model.image_encoder,
            device_ids=[device.index],
            output_device=device.index,
        )
    _set_seed(task_args.seed, rank_offset=True)

    states = []
    for head_idx, name in enumerate(local_datasets):
        loader = _build_loader(task_args, model, name)
        states.append(TaskState(name, model, [], loader, iter(loader), head_idx=head_idx))
    model.train()
    params = [param for param in model.image_encoder.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=task_args.lr, weight_decay=task_args.wd)
    scheduler = cosine_lr(
        optimizer, task_args.lr, task_args.warmup_length, task_args.num_steps
    )
    optimizer.zero_grad(set_to_none=True)
    loss_fn = _loss_function(task_args)

    def prepare():
        return _prepare_batches(states, task_args, device)

    def train_step(batches, step: int, sanity: bool = False):
        losses = []
        flat_batches = [
            (state, microbatch)
            for state, task_batches in zip(states, batches)
            for microbatch in task_batches
        ]
        for batch_idx, (state, (cpu_inputs, cpu_labels)) in enumerate(flat_batches):
            should_sync = batch_idx == len(flat_batches) - 1
            sync_context = contextlib.nullcontext()
            if is_distributed() and not should_sync:
                sync_context = model.image_encoder.no_sync()
            with sync_context:
                inputs = cpu_inputs.to(device)
                labels = cpu_labels.to(device)
                loss = loss_fn(model(inputs, state.head_idx), labels)
                # DDP averages rank gradients. This scaling yields the exact
                # global mean over tasks even when shards have unequal sizes.
                scale = (
                    get_world_size() / (total_tasks * task_args.num_grad_accumulation)
                    if is_distributed()
                    else 1.0 / (total_tasks * task_args.num_grad_accumulation)
                )
                (loss * scale).backward()
                losses.append(loss.detach())
        gradients_present = _has_gradient(params)
        scheduler(step)
        if task_args.clip_mode != "noclip":
            torch.nn.utils.clip_grad_norm_(params, task_args.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return {
            "finite_loss": _finite_loss(losses) if sanity else True,
            "gradients_present": gradients_present,
            "optimizer_steps": 1,
            "extra_operations": 0,
            "task_minibatches": len(flat_batches),
        }

    return MethodRuntime(states, prepare, train_step, sum(param.numel() for param in params))


def _build_runtime(
    method: str,
    task_args: argparse.Namespace,
    datasets: list[str],
    device: torch.device,
) -> MethodRuntime:
    local_datasets = datasets[get_rank() :: get_world_size()]
    if not local_datasets:
        raise RuntimeError("number of distributed processes cannot exceed number of tasks")
    if method == "socoft":
        return _build_socoft_runtime(task_args, local_datasets, len(datasets), device)
    if method == "hard_mtl":
        return _build_hard_mtl_runtime(task_args, local_datasets, device, len(datasets))
    return _build_specialist_runtime(method, task_args, local_datasets, len(datasets), device)


def _aggregate_sanity(
    local: dict,
    method: str,
    total_tasks: int,
    num_grad_accumulation: int,
    device: torch.device,
) -> dict:
    finite = _global_all(local["finite_loss"], device)
    gradients = _global_all(local["gradients_present"], device)
    optimizer_steps = _global_sum_int(local["optimizer_steps"], device)
    extra_operations = _global_sum_int(local["extra_operations"], device)
    task_minibatches = _global_sum_int(local["task_minibatches"], device)
    expected_steps = total_tasks if method not in {"socoft", "hard_mtl"} else get_world_size()
    expected_minibatches = total_tasks * num_grad_accumulation
    result = {
        "loss_finite": finite,
        "parameters_received_gradients": gradients,
        "optimizer_step_executed": optimizer_steps == expected_steps,
        "optimizer_step_calls_across_ranks": optimizer_steps,
        "expected_optimizer_step_calls": expected_steps,
        "equivalent_task_data_exposure": task_minibatches == expected_minibatches,
        "task_minibatches_per_global_step": task_minibatches,
        "expected_task_minibatches_per_global_step": expected_minibatches,
        "method_specific_operation_executed": True,
    }
    if method in {"saft", "mergopt"}:
        result["method_specific_operation_executed"] = extra_operations == total_tasks
        result["method_specific_operation"] = (
            "ASAM second forward/backward pass"
            if method == "saft"
            else "merge-offset perturb/closure/restore"
        )
    elif method == "socoft":
        # This first sanity step is coupled for the normal/default tau=1. For
        # larger tau, the final counter is checked after all warm-up steps.
        result["method_specific_operation_executed"] = extra_operations > 0
        result["method_specific_operation"] = "training-time merge/coupling gradient"
    return result


def _gpu_peaks(device: torch.device) -> tuple[list[float], list[float]]:
    gib = 1024.0**3
    allocated = _per_gpu_values(torch.cuda.max_memory_allocated(device) / gib, device)
    reserved = _per_gpu_values(torch.cuda.max_memory_reserved(device) / gib, device)
    return allocated, reserved


def _failed_sanity_checks(sanity: dict, include_method_specific: bool = True) -> list[str]:
    checked_names = {
        "loss_finite",
        "parameters_received_gradients",
        "optimizer_step_executed",
        "equivalent_task_data_exposure",
    }
    if include_method_specific:
        checked_names.add("method_specific_operation_executed")
    return [name for name in checked_names if not sanity[name]]


def _complete_sanity_metadata(sanity: dict) -> None:
    sanity.update(
        {
            "batches_prefetched_before_timing": True,
            "peak_memory_reset_after_warmup": True,
            "cuda_synchronization_surrounds_measured_region": True,
        }
    )


def _print_sanity() -> None:
    if is_main_process():
        print(
            "  sanity: finite loss=yes, gradients=yes, optimizer.step=yes, "
            "equal task exposure=yes, method-specific computation=yes, "
            "post-warm-up memory reset=yes, synchronized timing=yes"
        )


def _result_record(
    method: str,
    architecture: str,
    args: argparse.Namespace,
    measured_times: list[float],
    allocated: list[float],
    reserved: list[float],
    trainable_parameter_count: int,
    sanity: dict,
    execution_strategy: str,
    execution_details: dict | None = None,
) -> dict:
    result = {
        "method": METHOD_LABELS[method],
        "method_key": method,
        "architecture": architecture,
        "mean_time_per_step_s": statistics.fmean(measured_times),
        "std_time_per_step_s": statistics.pstdev(measured_times),
        "median_time_per_step_s": statistics.median(measured_times),
        "relative_time": None,
        "peak_vram_gb": max(allocated),
        "per_gpu_peak_vram_gb": allocated,
        "per_gpu_peak_reserved_gb": reserved,
        "num_gpus": get_world_size(),
        "warmup_steps": args.benchmark_warmup_steps,
        "measured_steps": args.benchmark_measure_steps,
        "raw_step_times_s": measured_times,
        "max_trainable_parameters_per_gpu_runtime": trainable_parameter_count,
        "execution_strategy": execution_strategy,
        "sanity_checks": sanity,
    }
    if execution_details is not None:
        result["execution_details"] = execution_details
    return result


def _benchmark_independent_waves(
    method: str,
    architecture: str,
    args: argparse.Namespace,
    datasets: list[str],
    device: torch.device,
) -> dict:
    """Benchmark independent specialists in waves of at most one task per GPU."""
    task_args = _architecture_args(args, architecture, device)
    world_size = get_world_size()
    wave_count = math.ceil(len(datasets) / world_size)
    measured_times = [0.0] * args.benchmark_measure_steps
    local_peak_allocated = 0.0
    local_peak_reserved = 0.0
    local_max_trainable = 0
    wave_details = []
    local_sanity = {
        "finite_loss": True,
        "gradients_present": True,
        "optimizer_steps": 0,
        "extra_operations": 0,
        "task_minibatches": 0,
    }

    if is_main_process():
        print(
            f"\n[{architecture}] {METHOD_LABELS[method]}: {len(datasets)} tasks, "
            f"{world_size} GPU(s), {wave_count} task wave(s)"
        )

    for wave_idx in range(wave_count):
        task_idx = wave_idx * world_size + get_rank()
        active = task_idx < len(datasets)
        runtime = None
        if active:
            # A task's seed is invariant to its GPU assignment and method.
            _set_seed(args.seed + task_idx, rank_offset=False)
            runtime = _build_specialist_runtime(
                method,
                task_args,
                [datasets[task_idx]],
                len(datasets),
                device,
            )
            local_max_trainable = max(
                local_max_trainable, runtime.trainable_parameter_count
            )

        for step in range(args.benchmark_warmup_steps):
            if not active:
                continue
            batches = runtime.prepare_batches()
            checks = runtime.train_step(batches, step, sanity=(step == 0))
            if step == 0:
                local_sanity["finite_loss"] &= checks["finite_loss"]
                local_sanity["gradients_present"] &= checks["gradients_present"]
                local_sanity["optimizer_steps"] += checks["optimizer_steps"]
                local_sanity["extra_operations"] += checks["extra_operations"]
                local_sanity["task_minibatches"] += checks["task_minibatches"]
            del batches

        # Each wave receives its own steady-state memory reset. This prevents a
        # previous specialist's allocator history from inflating the next one.
        _synchronize_workers(device)
        torch.cuda.reset_peak_memory_stats(device)
        wave_times = []
        for measured_idx in range(args.benchmark_measure_steps):
            batches = runtime.prepare_batches() if active else None
            _synchronize_workers(device)
            started = perf_counter()
            if active:
                runtime.train_step(
                    batches,
                    args.benchmark_warmup_steps + measured_idx,
                    sanity=False,
                )
            _synchronize_workers(device)
            wave_elapsed = _global_max(perf_counter() - started, device)
            wave_times.append(wave_elapsed)
            measured_times[measured_idx] += wave_elapsed
            del batches

        _synchronize_workers(device)
        gib = 1024.0**3
        wave_local_allocated = torch.cuda.max_memory_allocated(device) / gib
        wave_local_reserved = torch.cuda.max_memory_reserved(device) / gib
        local_peak_allocated = max(local_peak_allocated, wave_local_allocated)
        local_peak_reserved = max(local_peak_reserved, wave_local_reserved)
        wave_allocated = _per_gpu_values(wave_local_allocated, device)
        wave_reserved = _per_gpu_values(wave_local_reserved, device)
        wave_details.append(
            {
                "wave": wave_idx + 1,
                "tasks_by_rank": [
                    datasets[index] if index < len(datasets) else None
                    for index in range(
                        wave_idx * world_size, (wave_idx + 1) * world_size
                    )
                ],
                "raw_step_times_s": wave_times,
                "per_gpu_peak_vram_gb": wave_allocated,
                "per_gpu_peak_reserved_gb": wave_reserved,
            }
        )

        # Loading the next wave is setup work, not part of a global step.
        _shutdown_runtime_data_workers(runtime)
        del runtime
        gc.collect()
        torch.cuda.empty_cache()
        _synchronize_workers(device)

    sanity = _aggregate_sanity(
        local_sanity,
        method,
        len(datasets),
        args.num_grad_accumulation,
        device,
    )
    failed_checks = _failed_sanity_checks(sanity)
    if failed_checks:
        raise RuntimeError(f"{METHOD_LABELS[method]} failed sanity checks: {failed_checks}")
    _complete_sanity_metadata(sanity)
    sanity["one_specialist_per_gpu"] = True
    sanity["wave_count"] = wave_count
    _print_sanity()

    allocated = _per_gpu_values(local_peak_allocated, device)
    reserved = _per_gpu_values(local_peak_reserved, device)
    max_trainable = int(_global_max(float(local_max_trainable), device))
    result = _result_record(
        method,
        architecture,
        args,
        measured_times,
        allocated,
        reserved,
        max_trainable,
        sanity,
        "one independent specialist per GPU; task-wave wall times summed",
        {
            "wave_count": wave_count,
            "global_step_aggregation": "elementwise sum of raw wave step times",
            "waves": wave_details,
        },
    )
    return result


def benchmark_method(
    method: str,
    architecture: str,
    args: argparse.Namespace,
    datasets: list[str],
    device: torch.device,
) -> dict:
    if method in INDEPENDENT_METHODS:
        return _benchmark_independent_waves(
            method, architecture, args, datasets, device
        )

    _set_seed(args.seed)
    task_args = _architecture_args(args, architecture, device)
    runtime = _build_runtime(method, task_args, datasets, device)

    if is_main_process():
        print(
            f"\n[{architecture}] {METHOD_LABELS[method]}: "
            f"{len(datasets)} tasks, {get_world_size()} GPU(s)"
        )

    sanity = None
    coupling_operations = 0
    for step in range(args.benchmark_warmup_steps):
        batches = runtime.prepare_batches()
        local_checks = runtime.train_step(batches, step, sanity=(step == 0))
        coupling_operations += local_checks["extra_operations"]
        if step == 0:
            sanity = _aggregate_sanity(
                local_checks,
                method,
                len(datasets),
                args.num_grad_accumulation,
                device,
            )
        del batches

    _synchronize_workers(device)
    failed_checks = _failed_sanity_checks(
        sanity, include_method_specific=(method != "socoft")
    )
    if failed_checks:
        raise RuntimeError(f"{METHOD_LABELS[method]} failed sanity checks: {failed_checks}")

    # The reset is deliberately after all warm-up work and immediately before
    # measured-step prefetch/compute. Current model allocations remain counted.
    _synchronize_workers(device)
    torch.cuda.reset_peak_memory_stats(device)
    measured_times = []
    for measured_idx in range(args.benchmark_measure_steps):
        global_step = args.benchmark_warmup_steps + measured_idx
        batches = runtime.prepare_batches()
        _synchronize_workers(device)
        started = perf_counter()
        measured_checks = runtime.train_step(batches, global_step, sanity=False)
        coupling_operations += measured_checks["extra_operations"]
        _synchronize_workers(device)
        local_elapsed = perf_counter() - started
        measured_times.append(_global_max(local_elapsed, device))
        del batches

    _synchronize_workers(device)
    allocated, reserved = _gpu_peaks(device)
    if method == "socoft":
        global_coupling_count = _global_sum_int(coupling_operations, device)
        sanity["method_specific_operation_executed"] = global_coupling_count > 0
        if not sanity["method_specific_operation_executed"]:
            raise RuntimeError(
                "SCouT did not execute a coupling update; reduce --coupling-tau "
                "or increase the benchmark step counts"
            )
    _complete_sanity_metadata(sanity)
    _print_sanity()
    result = _result_record(
        method,
        architecture,
        args,
        measured_times,
        allocated,
        reserved,
        runtime.trainable_parameter_count,
        sanity,
        (
            "coupled round-robin task shards"
            if method == "socoft"
            else "DDP shared encoder with round-robin task shards"
        ),
    )
    _shutdown_runtime_data_workers(runtime)
    return result


def _hardware_metadata(device: torch.device) -> list[dict]:
    local = {
        "rank": get_rank(),
        "logical_cuda_device": device.index,
        "name": torch.cuda.get_device_name(device),
        "total_memory_gb": torch.cuda.get_device_properties(device).total_memory / 1024.0**3,
    }
    if not is_distributed():
        return [local]
    gathered = [None for _ in range(get_world_size())]
    torch.distributed.all_gather_object(gathered, local)
    return gathered


def _csv_rows(results: list[dict], world_size: int) -> tuple[list[str], list[dict]]:
    fields = [
        "method",
        "architecture",
        "mean_time_per_step_s",
        "std_time_per_step_s",
        "median_time_per_step_s",
        "relative_time",
        "peak_vram_gb",
    ]
    fields += [f"gpu{idx}_peak_vram_gb" for idx in range(world_size)]
    fields += [f"gpu{idx}_peak_reserved_gb" for idx in range(world_size)]
    fields += ["num_gpus", "warmup_steps", "measured_steps", "execution_strategy"]
    rows = []
    for result in results:
        row = {field: result.get(field) for field in fields}
        for idx, peak in enumerate(result["per_gpu_peak_vram_gb"]):
            row[f"gpu{idx}_peak_vram_gb"] = peak
        for idx, peak in enumerate(result["per_gpu_peak_reserved_gb"]):
            row[f"gpu{idx}_peak_reserved_gb"] = peak
        rows.append(row)
    return fields, rows


def _write_outputs(payload: dict, output_dir: Path, prefix: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{prefix}.json"
    csv_path = output_dir / f"{prefix}.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    fields, rows = _csv_rows(payload["results"], payload["configuration"]["num_gpus"])
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path, json_path


def _print_summary(results: list[dict], architectures: list[str]) -> None:
    for architecture in architectures:
        rows = [row for row in results if row["architecture"] == architecture]
        if not rows:
            continue
        print("\n" + "=" * 67)
        print(architecture.replace("ViT-B-32", "ViT-B/32").replace("ViT-L-14", "ViT-L/14"))
        print("=" * 67)
        print(f"{'Method':<18}{'Time/step (s)':>16}{'Rel. time':>13}{'Peak VRAM (GB)':>20}")
        for row in rows:
            relative = "n/a" if row["relative_time"] is None else f"{row['relative_time']:.2f}x"
            print(
                f"{row['method']:<18}{row['mean_time_per_step_s']:>16.3f}"
                f"{relative:>13}{row['peak_vram_gb']:>20.1f}"
            )


def _normalize_relative_times(results: list[dict]) -> None:
    baselines = {
        result["architecture"]: result["mean_time_per_step_s"]
        for result in results
        if result["method_key"] == "independent_ft"
    }
    for result in results:
        baseline = baselines.get(result["architecture"])
        result["relative_time"] = (
            result["mean_time_per_step_s"] / baseline if baseline else None
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("training-efficiency timing and VRAM measurement require CUDA")

    distributed_started = setup_torchrun()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    datasets = _normalize_datasets(args.train_dataset)
    if get_world_size() > len(datasets):
        raise ValueError("number of GPUs/processes cannot exceed number of benchmark tasks")

    try:
        hardware = _hardware_metadata(device)
        output_dir = Path(args.output_dir)
        if is_main_process():
            output_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"Benchmarking {len(datasets)} tasks, batch size {args.batch_size}, "
                f"warm-up/measured steps {args.benchmark_warmup_steps}/"
                f"{args.benchmark_measure_steps}, precision fp32 (AMP disabled)."
            )
        if is_distributed():
            torch.distributed.barrier()

        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": {
                "architectures": args.architectures,
                "methods": [METHOD_LABELS[name] for name in args.methods],
                "method_implementation_mapping": {
                    "Independent FT": "src.indep_finetune standard AdamW path",
                    "FTTS": "src.linearize.LinearizedImageEncoder (repository TFT/linear mode)",
                    "FT-Attention": "src.attention_ft.configure_attention_finetuning",
                    "SAFT": "src.sam.SAM wrapping AdamW",
                    "MergOPT": "src.mergopt.MergOPT wrapping AdamW",
                    "SCouT": "bucketed analytic coupling-gradient helper",
                    "Hard MTL": "shared encoder and task heads as in src.hard_joint_finetune",
                },
                "datasets": datasets,
                "batch_size": args.batch_size,
                "num_tasks": len(datasets),
                "num_grad_accumulation": args.num_grad_accumulation,
                "warmup_steps": args.benchmark_warmup_steps,
                "measured_steps": args.benchmark_measure_steps,
                "seed": args.seed,
                "precision": "fp32",
                "amp_enabled": False,
                "optimizer": "AdamW",
                "learning_rate": args.lr,
                "weight_decay": args.wd,
                "clip_mode": args.clip_mode,
                "num_gpus": get_world_size(),
                "resource_policy": (
                    "fixed GPU count with one specialist per GPU when tasks "
                    "are independent"
                ),
                "task_distribution": {
                    "independent_methods": (
                        "waves of at most one specialist per GPU; wave wall times "
                        "are summed"
                    ),
                    "SCouT": "all task specialists persist in round-robin GPU shards",
                    "Hard MTL": "DDP shared-encoder replicas with round-robin task shards",
                },
                "independent_global_step_time": (
                    "sum over task waves of max per-rank measured step time"
                ),
                "peak_vram_definition": (
                    "maximum allocated GiB over devices and, for independent "
                    "methods, over task waves"
                ),
                "timing_std_definition": "population standard deviation",
                "dataloader_retrieval_timed": False,
                "host_to_device_transfer_timed": True,
                "cuda_synchronization_surrounds_each_measured_step": True,
                "peak_memory_reset_after_warmup": True,
            },
            "software": {
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            },
            "gpus": hardware,
            "results": [],
        }

        for architecture in args.architectures:
            for method in args.methods:
                result = benchmark_method(method, architecture, args, datasets, device)
                payload["results"].append(result)
                _normalize_relative_times(payload["results"])
                if is_main_process():
                    _write_outputs(payload, output_dir, args.output_prefix)
                    print(
                        f"  measured: {result['mean_time_per_step_s']:.3f} s/step, "
                        f"peak {result['peak_vram_gb']:.1f} GiB"
                    )

                # Release this method before constructing the next one. This is
                # outside both warm-up and measured regions.
                del result
                gc.collect()
                torch.cuda.empty_cache()
                _synchronize_workers(device)

        _normalize_relative_times(payload["results"])
        if is_main_process():
            csv_path, json_path = _write_outputs(payload, output_dir, args.output_prefix)
            _print_summary(payload["results"], args.architectures)
            print(f"\nCSV:  {csv_path}")
            print(f"JSON: {json_path}")
        return 0
    finally:
        if distributed_started:
            cleanup_ddp()


if __name__ == "__main__":
    raise SystemExit(main())
