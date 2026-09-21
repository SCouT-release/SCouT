#!/usr/bin/env python3
"""Evaluate and plot two-task SCouT loss landscapes."""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import utils  # noqa: E402
from src.datasets.common import maybe_dictionarize  # noqa: E402
from src.datasets.registry import get_dataset  # noqa: E402
from src.heads import get_classification_head  # noqa: E402

DEFAULT_TASKS = ("Cars", "RESISC45")
DEFAULT_LAMBDAS = (0.0, 0.5, 5.0)


def legend_arrow(
    legend, orig_handle, xdescent, ydescent, width, height, fontsize
) -> FancyArrowPatch:
    del legend, orig_handle, fontsize
    return FancyArrowPatch(
        (xdescent, ydescent + 0.5 * height),
        (xdescent + width, ydescent + 0.5 * height),
        arrowstyle="->",
        mutation_scale=9,
        linewidth=1.15,
        color="white",
        path_effects=[
            path_effects.Stroke(linewidth=2.2, foreground="black"),
            path_effects.Normal(),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-lambda 2D loss landscapes for two SCouT specialists."
    )
    parser.add_argument("--tasks", nargs=2, default=DEFAULT_TASKS, metavar=("TASK1", "TASK2"))
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/ViT-B-32"))
    parser.add_argument(
        "--pretrained-pattern",
        default="{task_val}/zeroshot.pt",
        help="Checkpoint path relative to --checkpoint-dir; supports {task} and {task_val}.",
    )
    parser.add_argument(
        "--independent-pattern",
        default="{task_val}/finetuned.pt",
        help="Specialist path for lambda=0; supports {task}, {task_val}, and {lambda}.",
    )
    parser.add_argument(
        "--scout-pattern",
        "--socoft-pattern",
        dest="scout_pattern",
        default="{task_val}/sj_finetuned_lambda_sweep_1_{lambda}.pt",
        help="Specialist path for lambda>0; supports {task}, {task_val}, and {lambda}.",
    )
    parser.add_argument("--data-location", default=os.path.expanduser("~/data"))
    parser.add_argument(
        "--openclip-cachedir",
        default=os.path.expanduser("~/openclip-cachedir/open_clip"),
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/landscape_sweep"))
    parser.add_argument("--grid-size", type=int, default=21)
    parser.add_argument("--subset-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--barrier-points", type=int, default=11)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument(
        "--min-orthogonal-ratios",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Minimum y-span/x-span ratio for each requested lambda, in the same "
            "order as --lambdas. One value applies to every lambda. Defaults to "
            "0 for lambda=0, 0.35 for 0<lambda<=0.5, and 0.15 for lambda>0.5."
        ),
    )
    parser.add_argument(
        "--plot-orthogonal-ratios",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Displayed y-span/x-span ratio for each requested lambda. Zero keeps "
            "the full evaluated y range. Defaults to full range for lambda=0, "
            "0.10 for 0<lambda<=0.5, and 0.04 for lambda>0.5."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use float16 autocast on CUDA (enabled by default).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Plot all available requested-lambda caches without loading models or data.",
    )
    args = parser.parse_args()

    if len(set(args.tasks)) != 2:
        parser.error("--tasks must contain two distinct task names")
    if any(value < 0 for value in args.lambdas):
        parser.error("--lambdas must be non-negative")
    if args.grid_size < 2:
        parser.error("--grid-size must be at least 2")
    if args.subset_size < 1 or args.batch_size < 1:
        parser.error("--subset-size and --batch-size must be positive")
    if args.barrier_points < 2:
        parser.error("--barrier-points must be at least 2")
    if args.margin <= 0:
        parser.error("--margin must be positive")
    if args.min_orthogonal_ratios is None:
        args.min_orthogonal_ratios = [
            0.0 if np.isclose(value, 0.0) else 0.35 if value <= 0.5 else 0.15
            for value in args.lambdas
        ]
    elif len(args.min_orthogonal_ratios) == 1:
        args.min_orthogonal_ratios *= len(args.lambdas)
    elif len(args.min_orthogonal_ratios) != len(args.lambdas):
        parser.error(
            "--min-orthogonal-ratios must contain one value or one value per lambda"
        )
    if any(value < 0 for value in args.min_orthogonal_ratios):
        parser.error("--min-orthogonal-ratios values must be non-negative")
    if args.plot_orthogonal_ratios is None:
        args.plot_orthogonal_ratios = [
            0.0 if np.isclose(value, 0.0) else 0.10 if value <= 0.5 else 0.04
            for value in args.lambdas
        ]
    elif len(args.plot_orthogonal_ratios) == 1:
        args.plot_orthogonal_ratios *= len(args.lambdas)
    elif len(args.plot_orthogonal_ratios) != len(args.lambdas):
        parser.error(
            "--plot-orthogonal-ratios must contain one value or one value per lambda"
        )
    if any(value < 0 for value in args.plot_orthogonal_ratios):
        parser.error("--plot-orthogonal-ratios values must be non-negative")
    return args


def lambda_tag(value: float) -> str:
    return f"{value:g}"


def task_name(task: str) -> str:
    return task[:-3] if task.endswith("Val") else task


def task_val_name(task: str) -> str:
    return task if task.endswith("Val") else f"{task}Val"


def pair_name(tasks: list[str] | tuple[str, ...]) -> str:
    return "_".join("".join(c if c.isalnum() else "_" for c in task_name(t)) for t in tasks)


def result_path(args: argparse.Namespace, coupling_lambda: float) -> Path:
    return args.output_dir / f"{pair_name(args.tasks)}_lambda_{lambda_tag(coupling_lambda)}.npz"


def resolve_checkpoint(
    args: argparse.Namespace,
    pattern: str,
    task: str,
    coupling_lambda: float,
) -> Path:
    rendered = pattern.format(
        **{
            "task": task_name(task),
            "task_val": task_val_name(task),
            "lambda": lambda_tag(coupling_lambda),
        }
    )
    path = Path(rendered).expanduser()
    return path if path.is_absolute() else args.checkpoint_dir / path


def specialist_paths(args: argparse.Namespace, coupling_lambda: float) -> list[Path]:
    pattern = (
        args.independent_pattern
        if np.isclose(coupling_lambda, 0.0)
        else args.scout_pattern
    )
    return [resolve_checkpoint(args, pattern, task, coupling_lambda) for task in args.tasks]


def load_encoder(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = utils.torch_load(str(path))
    if not hasattr(model, "named_parameters") or not hasattr(model, "val_preprocess"):
        raise TypeError(
            f"Expected an ImageEncoder checkpoint at {path}, got {type(model).__name__}"
        )
    return model


def trainable_parameters(model, expected_names: list[str] | None = None) -> tuple[list[str], list]:
    named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    names = [name for name, _ in named]
    if expected_names is not None and names != expected_names:
        missing = sorted(set(expected_names) - set(names))
        extra = sorted(set(names) - set(expected_names))
        raise ValueError(
            "Checkpoint parameter names/order do not match the pretrained model. "
            f"Missing: {missing[:3]}; extra: {extra[:3]}"
        )
    return names, [param for _, param in named]


def check_shapes(reference: list[torch.Tensor], candidate: list[torch.Tensor], path: Path) -> None:
    for index, (base, other) in enumerate(zip(reference, candidate)):
        if base.shape != other.shape:
            raise ValueError(
                f"Parameter {index} in {path} has shape {tuple(other.shape)}, "
                f"expected {tuple(base.shape)}"
            )


def vector_dot(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    return sum(
        torch.sum(a.detach() * b.detach(), dtype=torch.float64).item()
        for a, b in zip(left, right)
    )


def vector_norm(vector: list[torch.Tensor]) -> float:
    return float(np.sqrt(max(vector_dot(vector, vector), 0.0)))


def verify_common_pretrained(
    args: argparse.Namespace,
    names: list[str],
    base_params: list[torch.Tensor],
) -> None:
    other_path = resolve_checkpoint(args, args.pretrained_pattern, args.tasks[1], 0.0)
    other = load_encoder(other_path)
    _, other_params = trainable_parameters(other, names)
    check_shapes(base_params, other_params, other_path)
    max_difference = max(
        (base - candidate.detach()).abs().max().item()
        for base, candidate in zip(base_params, other_params)
    )
    del other
    if max_difference > 1e-6:
        raise ValueError(
            "The two pretrained checkpoints are not a common initialization: "
            f"maximum parameter difference is {max_difference:.3e}."
        )
    print(f"Common pretrained checkpoint verified (max difference {max_difference:.2e}).")


def build_basis(
    paths: list[Path],
    names: list[str],
    base_params: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, float], dict[str, np.ndarray]]:
    specialist1 = load_encoder(paths[0])
    _, tau1 = trainable_parameters(specialist1, names)
    check_shapes(base_params, tau1, paths[0])
    with torch.no_grad():
        for value, base in zip(tau1, base_params):
            value.sub_(base)
    norm1 = vector_norm(tau1)
    if norm1 <= 1e-12:
        raise ValueError(f"First specialist has a zero task vector: {paths[0]}")

    reconstruction1_sq = 0.0
    reconstruction1_max = 0.0
    with torch.no_grad():
        for value in tau1:
            original = value.detach().clone()
            value.div_(norm1)
            error = value * norm1 - original
            reconstruction1_sq += torch.sum(error * error, dtype=torch.float64).item()
            reconstruction1_max = max(reconstruction1_max, error.abs().max().item())

    specialist2 = load_encoder(paths[1])
    _, tau2 = trainable_parameters(specialist2, names)
    check_shapes(base_params, tau2, paths[1])
    with torch.no_grad():
        for value, base in zip(tau2, base_params):
            value.sub_(base)
    norm2 = vector_norm(tau2)
    projection = vector_dot(tau2, tau1)
    with torch.no_grad():
        for value, u_value in zip(tau2, tau1):
            value.add_(u_value, alpha=-projection)
    residual_norm = vector_norm(tau2)
    if residual_norm <= 1e-12:
        raise ValueError("The two task vectors are collinear; a 2D plane cannot be constructed.")

    reconstruction2_sq = 0.0
    reconstruction2_max = 0.0
    with torch.no_grad():
        for value, u_value in zip(tau2, tau1):
            original = value.detach().clone().add_(u_value, alpha=projection)
            value.div_(residual_norm)
            reconstructed = value * residual_norm + u_value * projection
            error = reconstructed - original
            reconstruction2_sq += torch.sum(error * error, dtype=torch.float64).item()
            reconstruction2_max = max(reconstruction2_max, error.abs().max().item())

    u_norm = vector_norm(tau1)
    v_norm = vector_norm(tau2)
    uv_dot = vector_dot(tau1, tau2)
    reconstruction1_relative = np.sqrt(reconstruction1_sq) / norm1
    reconstruction2_relative = np.sqrt(reconstruction2_sq) / norm2
    basis_tolerance = 5e-5
    if (
        abs(u_norm - 1.0) > basis_tolerance
        or abs(v_norm - 1.0) > basis_tolerance
        or abs(uv_dot) > basis_tolerance
    ):
        raise RuntimeError(
            f"Invalid basis: ||u||={u_norm:.8f}, ||v||={v_norm:.8f}, u.v={uv_dot:.3e}"
        )
    if max(reconstruction1_relative, reconstruction2_relative) > 1e-5:
        raise RuntimeError(
            "Specialist reconstruction check failed: "
            f"relative errors are {reconstruction1_relative:.3e} and "
            f"{reconstruction2_relative:.3e}."
        )

    specialist1_coord = np.array([norm1, 0.0], dtype=np.float64)
    specialist2_coord = np.array([projection, residual_norm], dtype=np.float64)
    coordinates = {
        "pretrained": np.zeros(2, dtype=np.float64),
        "specialist_0": specialist1_coord,
        "specialist_1": specialist2_coord,
        "merge": 0.5 * (specialist1_coord + specialist2_coord),
    }
    distance = 0.25 * (norm1**2 + norm2**2 - 2.0 * norm1 * projection)
    distance_scale = 0.5 * (norm1**2 + norm2**2)
    diagnostics = {
        "basis_u_norm": u_norm,
        "basis_v_norm": v_norm,
        "basis_dot": uv_dot,
        "task_vector_norm_0": norm1,
        "task_vector_norm_1": norm2,
        "reconstruction_relative_error_0": reconstruction1_relative,
        "reconstruction_relative_error_1": reconstruction2_relative,
        "reconstruction_max_error_0": reconstruction1_max,
        "reconstruction_max_error_1": reconstruction2_max,
        "specialist_merge_distance": distance,
        "normalized_specialist_merge_distance": distance / distance_scale,
    }
    print(
        "Basis checks: "
        f"||u||={u_norm:.8f}, ||v||={v_norm:.8f}, u.v={uv_dot:.2e}; "
        f"reconstruction={reconstruction1_relative:.2e}/{reconstruction2_relative:.2e}"
    )
    return tau1, tau2, diagnostics, coordinates


def coordinate_grid(
    coordinates: dict[str, np.ndarray],
    grid_size: int,
    margin: float,
    min_orthogonal_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.stack(list(coordinates.values()))
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = 0.5 * (low + high)

    x_span = max(float(high[0] - low[0]), 1e-12)
    natural_y_span = float(high[1] - low[1])
    y_span = max(natural_y_span, min_orthogonal_ratio * x_span, 1e-6 * x_span)

    x_half_range = (0.5 + margin) * x_span
    y_half_range = (0.5 + margin) * y_span
    return (
        np.linspace(
            center[0] - x_half_range,
            center[0] + x_half_range,
            grid_size,
            dtype=np.float64,
        ),
        np.linspace(
            center[1] - y_half_range,
            center[1] + y_half_range,
            grid_size,
            dtype=np.float64,
        ),
    )


def prepare_task_data(
    args: argparse.Namespace, image_encoder
) -> tuple[list, list, list[np.ndarray]]:
    heads = []
    all_batches = []
    all_indices = []
    pin_memory = torch.device(args.device).type == "cuda"

    for task in args.tasks:
        head = get_classification_head(args, task).to(args.device).eval()
        dataset = get_dataset(
            task_val_name(task),
            image_encoder.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
        validation_set = dataset.test_dataset
        generator = np.random.default_rng(args.seed)
        count = min(args.subset_size, len(validation_set))
        indices = np.sort(generator.choice(len(validation_set), size=count, replace=False))
        loader = DataLoader(
            Subset(validation_set, indices.tolist()),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=pin_memory,
        )
        batches = []
        for batch in tqdm(loader, desc=f"Caching {task_val_name(task)} subset", leave=False):
            batch = maybe_dictionarize(batch)
            batches.append((batch["images"], batch["labels"]))
        heads.append(head)
        all_batches.append(batches)
        all_indices.append(indices)
        print(f"Using {count} deterministic validation examples for {task}.")
    return heads, all_batches, all_indices


def set_plane_point(
    model_params: list[torch.Tensor],
    base_params: list[torch.Tensor],
    u: list[torch.Tensor],
    v: list[torch.Tensor],
    x: float,
    y: float,
) -> None:
    with torch.no_grad():
        for parameter, base, u_value, v_value in zip(model_params, base_params, u, v):
            parameter.copy_(base)
            parameter.add_(u_value, alpha=float(x))
            parameter.add_(v_value, alpha=float(y))


def evaluate_loss(
    image_encoder,
    head,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    use_amp: bool,
) -> float:
    total_loss = 0.0
    total_examples = 0
    with torch.inference_mode():
        for images, labels in batches:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = head(image_encoder(images))
            total_loss += F.cross_entropy(logits.float(), labels, reduction="sum").item()
            total_examples += labels.numel()
    return total_loss / total_examples


def evaluate_grid(
    image_encoder,
    model_params: list[torch.Tensor],
    base_params: list[torch.Tensor],
    u: list[torch.Tensor],
    v: list[torch.Tensor],
    x_values: np.ndarray,
    y_values: np.ndarray,
    heads: list,
    batches: list,
    device: torch.device,
    use_amp: bool,
    coupling_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    losses = [np.empty((len(y_values), len(x_values)), dtype=np.float64) for _ in heads]
    progress = tqdm(
        total=len(x_values) * len(y_values),
        desc=f"lambda={lambda_tag(coupling_lambda)}",
    )
    for row, y_value in enumerate(y_values):
        for column, x_value in enumerate(x_values):
            set_plane_point(model_params, base_params, u, v, x_value, y_value)
            for task_index in range(len(heads)):
                losses[task_index][row, column] = evaluate_loss(
                    image_encoder,
                    heads[task_index],
                    batches[task_index],
                    device,
                    use_amp,
                )
            progress.update()
    progress.close()
    return losses[0], losses[1]


def evaluate_barriers(
    image_encoder,
    model_params: list[torch.Tensor],
    base_params: list[torch.Tensor],
    u: list[torch.Tensor],
    v: list[torch.Tensor],
    coordinates: dict[str, np.ndarray],
    heads: list,
    batches: list,
    device: torch.device,
    use_amp: bool,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_values = np.linspace(0.0, 1.0, num_points, dtype=np.float64)
    barrier_losses = []
    for task_index in range(2):
        start = coordinates[f"specialist_{task_index}"]
        end = coordinates["merge"]
        task_losses = np.empty(num_points, dtype=np.float64)
        for index, t_value in enumerate(t_values):
            point = (1.0 - t_value) * start + t_value * end
            set_plane_point(model_params, base_params, u, v, point[0], point[1])
            task_losses[index] = evaluate_loss(
                image_encoder,
                heads[task_index],
                batches[task_index],
                device,
                use_amp,
            )
        barrier_losses.append(task_losses)
    barriers = np.array(
        [
            max(0.0, values.max() - max(values[0], values[-1]))
            for values in barrier_losses
        ],
        dtype=np.float64,
    )
    return t_values, np.stack(barrier_losses), barriers


def save_result(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    print(f"Saved landscape cache: {path}")


def load_result(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def available_results(args: argparse.Namespace) -> list[tuple[Path, dict[str, np.ndarray]]]:
    results = []
    for coupling_lambda in args.lambdas:
        path = result_path(args, coupling_lambda)
        if path.is_file():
            results.append((path, load_result(path)))
    return results


def write_diagnostics(args: argparse.Namespace) -> Path | None:
    results = available_results(args)
    if not results:
        return None
    task_labels = [str(value) for value in results[0][1]["task_names"]]
    metrics = [
        ("Basis", "u norm", "basis_u_norm", "Should be close to 1"),
        ("Basis", "v norm", "basis_v_norm", "Should be close to 1"),
        ("Basis", "u-v dot product", "basis_dot", "Should be close to 0"),
        (
            "Task vectors",
            f"{task_labels[0]} task-vector norm",
            "task_vector_norm_0",
            "Distance moved from the pretrained model",
        ),
        (
            "Task vectors",
            f"{task_labels[1]} task-vector norm",
            "task_vector_norm_1",
            "Distance moved from the pretrained model",
        ),
        (
            "Reconstruction",
            f"{task_labels[0]} relative error",
            "reconstruction_relative_error_0",
            "Should be close to 0",
        ),
        (
            "Reconstruction",
            f"{task_labels[1]} relative error",
            "reconstruction_relative_error_1",
            "Should be close to 0",
        ),
        (
            "Reconstruction",
            f"{task_labels[0]} maximum error",
            "reconstruction_max_error_0",
            "Should be close to 0",
        ),
        (
            "Reconstruction",
            f"{task_labels[1]} maximum error",
            "reconstruction_max_error_1",
            "Should be close to 0",
        ),
        (
            "Merge geometry",
            "Specialist-to-merge distance",
            "specialist_merge_distance",
            "Lower means the specialists are closer",
        ),
        (
            "Merge geometry",
            "Normalized specialist-to-merge distance",
            "normalized_specialist_merge_distance",
            "Scale-free; lower means better task-vector alignment",
        ),
        (
            "Loss barrier",
            f"{task_labels[0]} interpolation barrier",
            "barrier_0",
            "Lower means a flatter path to the merge",
        ),
        (
            "Loss barrier",
            f"{task_labels[1]} interpolation barrier",
            "barrier_1",
            "Lower means a flatter path to the merge",
        ),
        (
            "Loss barrier",
            "Average interpolation barrier",
            "barrier_average",
            "Mean of the two task barriers",
        ),
    ]
    lambda_columns = [f"lambda={float(result['lambda']):g}" for _, result in results]
    fields = ["category", "metric", "metric_key", "interpretation", *lambda_columns]
    output = args.output_dir / f"{pair_name(args.tasks)}_diagnostics.csv"
    temporary = output.with_suffix(".tmp.csv")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for category, metric, key, interpretation in metrics:
            row = {
                "category": category,
                "metric": metric,
                "metric_key": key,
                "interpretation": interpretation,
            }
            for column, (_, result) in zip(lambda_columns, results):
                row[column] = f"{float(result[key]):.10g}"
            writer.writerow(row)
    temporary.replace(output)
    return output


def best_task_arithmetic(
    result: dict[str, np.ndarray],
) -> tuple[np.ndarray, tuple[float, float, float]]:
    row, column = np.unravel_index(
        np.nanargmin(result["average_loss"]), result["average_loss"].shape
    )
    coordinate = np.array([result["x"][column], result["y"][row]])
    task_0_loss = float(result["task_0_loss"][row, column])
    task_1_loss = float(result["task_1_loss"][row, column])
    return coordinate, (
        task_0_loss,
        task_1_loss,
        0.5 * (task_0_loss + task_1_loss),
    )


def contour_limits(
    results: list[dict[str, np.ndarray]],
    key: str,
    y_limits: list[tuple[float, float]],
    low_override: float | None = None,
) -> tuple[float, float]:
    visible_values = []
    for result, (y_min, y_max) in zip(results, y_limits):
        visible_rows = (result["y"] >= y_min) & (result["y"] <= y_max)
        visible_values.append(result[key][visible_rows, :].ravel())
    values = np.concatenate(visible_values)
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError(f"No finite values found for {key}")
    low = float(finite.min()) if low_override is None else low_override
    high = float(np.percentile(finite, 90.0))
    if np.isclose(low, high):
        high = low + 1e-6
    return low, high


def displayed_y_limits(
    result: dict[str, np.ndarray], plot_ratio: float, zero_fraction: float
) -> tuple[float, float]:
    y_values = result["y"]
    if plot_ratio == 0:
        return float(y_values[0]), float(y_values[-1])

    best_ta_coord, _ = best_task_arithmetic(result)
    coordinates = (
        result["pretrained_coord"],
        result["specialist_0_coord"],
        result["specialist_1_coord"],
        best_ta_coord,
    )
    positive_needed = 1.1 * max(0.0, *(float(point[1]) for point in coordinates))
    negative_needed = 1.1 * max(0.0, *(-float(point[1]) for point in coordinates))
    span = max(
        plot_ratio * float(np.ptp(result["x"])),
        positive_needed / (1.0 - zero_fraction),
        negative_needed / zero_fraction,
    )
    y_min = -zero_fraction * span
    y_max = (1.0 - zero_fraction) * span
    return max(float(y_values[0]), y_min), min(float(y_values[-1]), y_max)


def plot_results(args: argparse.Namespace) -> tuple[Path, Path] | None:
    loaded = available_results(args)
    if not loaded:
        print(f"No landscape caches found in {args.output_dir}; nothing to plot.")
        return None
    results = [result for _, result in loaded]
    plot_ratios = {
        coupling_lambda: ratio
        for coupling_lambda, ratio in zip(
            args.lambdas, args.plot_orthogonal_ratios
        )
    }
    task_labels = [str(value) for value in results[0]["task_names"]]
    row_keys = ("task_0_loss", "task_1_loss", "average_loss")
    row_labels = (f"{task_labels[0]} loss", f"{task_labels[1]} loss", "Average loss")
    baseline_result = next(
        (result for result in results if np.isclose(float(result["lambda"]), 0.0)),
        None,
    )
    baseline_losses = (
        best_task_arithmetic(baseline_result)[1]
        if baseline_result is not None
        else None
    )
    reference_y = baseline_result["y"] if baseline_result is not None else results[0]["y"]
    zero_fraction = float(-reference_y[0] / np.ptp(reference_y))
    display_y_limits = []
    for result in results:
        coupling_lambda = float(result["lambda"])
        plot_ratio = next(
            ratio
            for value, ratio in plot_ratios.items()
            if np.isclose(value, coupling_lambda)
        )
        display_y_limits.append(
            displayed_y_limits(result, plot_ratio, zero_fraction)
        )

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    columns = len(results)
    fig, axes = plt.subplots(
        3,
        columns,
        squeeze=False,
        figsize=(3.55 * columns + 0.65, 6.6),
        constrained_layout=True,
    )
    viridis = mpl.colormaps["viridis"]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "deep_purple_viridis",
        [
            (0.00, "#2A084A"),
            (0.15, "#34105C"),
            (0.28, viridis(0.05)),
            (0.38, viridis(0.18)),
            (0.55, viridis(0.42)),
            (0.75, viridis(0.68)),
            (1.00, viridis(1.00)),
        ],
    )
    for row, (key, row_label) in enumerate(zip(row_keys, row_labels)):
        baseline_loss = baseline_losses[row] if baseline_losses is not None else None
        specialist_floors = (
            [float(result["barrier_losses"][row, 0]) for result in results]
            if row < 2
            else None
        )
        low_override = min(specialist_floors) if specialist_floors else None
        low, high = contour_limits(results, key, display_y_limits, low_override)
        level_count = 25 if row < 2 else 33
        level_power = 4 if row < 2 else 5
        level_positions = np.linspace(0.0, 1.0, level_count)
        levels = low + (high - low) * level_positions**level_power
        norm = mpl.colors.PowerNorm(gamma=0.45, vmin=low, vmax=high)
        row_contour = None
        for column, result in enumerate(results):
            axis = axes[row, column]
            x_values = result["x"]
            y_values = result["y"]
            best_ta_coord, best_ta_losses = best_task_arithmetic(result)
            y_min, y_max = display_y_limits[column]
            panel_low = specialist_floors[column] if specialist_floors else low
            surface = np.clip(result[key], panel_low, high)
            row_contour = axis.contourf(
                x_values,
                y_values,
                surface,
                levels=levels,
                cmap=cmap,
                norm=norm,
            )
            axis.contour(
                x_values,
                y_values,
                surface,
                levels=levels[2::2],
                colors="black",
                linewidths=0.34,
                alpha=0.42,
            )
            if (
                baseline_loss is not None
                and result[key].min() < baseline_loss < result[key].max()
            ):
                axis.contour(
                    x_values,
                    y_values,
                    result[key],
                    levels=[baseline_loss],
                    colors="white",
                    linestyles="dotted",
                    linewidths=1.5,
                    zorder=4,
                )
            pretrained = result["pretrained_coord"]
            for endpoint_key in ("specialist_0_coord", "specialist_1_coord"):
                endpoint = result[endpoint_key]
                axis.annotate(
                    "",
                    xy=endpoint,
                    xytext=pretrained,
                    arrowprops={
                        "arrowstyle": "->",
                        "color": "white",
                        "linestyle": (0, (4, 3)),
                        "linewidth": 1.15,
                        "mutation_scale": 9,
                        "shrinkA": 5,
                        "shrinkB": 5,
                        "path_effects": [
                            path_effects.Stroke(linewidth=2.2, foreground="black"),
                            path_effects.Normal(),
                        ],
                    },
                    zorder=4,
                )
            markers = (
                (result["pretrained_coord"], "x", "black", 42, 1.5),
                (result["specialist_0_coord"], "x", "#F28E00", 48, 1.5),
                (result["specialist_1_coord"], "x", "#E31A1C", 48, 1.5),
                (best_ta_coord, "*", "#A000A0", 72, 0.8),
            )
            for point, marker, color, size, linewidth in markers:
                edge_kwargs = {"edgecolor": "white"} if marker == "*" else {}
                axis.scatter(
                    point[0],
                    point[1],
                    marker=marker,
                    s=size,
                    color=color,
                    linewidth=linewidth,
                    zorder=5,
                    **edge_kwargs,
                )
            if row == 2:
                current_merge_loss = best_ta_losses[2]
                axis.annotate(
                    f"Loss: {current_merge_loss:.3f}",
                    xy=best_ta_coord,
                    xytext=(0, 16),
                    textcoords="offset points",
                    horizontalalignment="center",
                    fontsize=7.2,
                    color="white",
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "black",
                        "edgecolor": "none",
                        "alpha": 0.68,
                    },
                    zorder=7,
                )
            axis.set_xlim(x_values[0], x_values[-1])
            axis.set_ylim(y_min, y_max)
            axis.set_aspect("auto")
            axis.xaxis.set_major_locator(MaxNLocator(5))
            axis.yaxis.set_major_locator(MaxNLocator(5))
            axis.tick_params(length=3, width=0.7)
            if row < 2:
                axis.tick_params(axis="x", bottom=False, labelbottom=False)
            if row == 0:
                axis.set_title(rf"$\lambda={float(result['lambda']):g}$", pad=5)
            if column == 0:
                axis.set_ylabel(f"{row_label}\n$y$")
            if row == 2:
                axis.set_xlabel("$x$")
        fig.colorbar(
            row_contour,
            ax=axes[row, :],
            pad=0.015,
            shrink=0.88,
            label="Cross-entropy (capped)",
            ticks=np.linspace(low, high, 4),
            format="%.2f",
        )

    legend_handles = [
        Line2D(
            [],
            [],
            marker="x",
            linestyle="none",
            color="black",
            markeredgewidth=1.5,
            label="Pretrained",
        ),
        Line2D(
            [],
            [],
            marker="x",
            linestyle="none",
            color="#F28E00",
            markeredgewidth=1.5,
            label=f"{task_labels[0]} specialist",
        ),
        Line2D(
            [],
            [],
            marker="x",
            linestyle="none",
            color="#E31A1C",
            markeredgewidth=1.5,
            label=f"{task_labels[1]} specialist",
        ),
        Line2D(
            [],
            [],
            marker="*",
            linestyle="none",
            markerfacecolor="#A000A0",
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=10,
            label="Best task arithmetic",
        ),
        FancyArrowPatch(
            (0, 0),
            (1, 0),
            arrowstyle="->",
            mutation_scale=9,
            linewidth=1.15,
            color="white",
            label="Task vectors",
            path_effects=[
                path_effects.Stroke(linewidth=2.2, foreground="black"),
                path_effects.Normal(),
            ],
        ),
        Line2D(
            [],
            [],
            color="0.35",
            linestyle=(0, (2, 2)),
            linewidth=1.5,
            label="Independent-FT best-TA loss",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        handler_map={FancyArrowPatch: HandlerPatch(patch_func=legend_arrow)},
        loc="outside upper center",
        ncol=3,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.25,
    )
    stem = args.output_dir / f"{pair_name(args.tasks)}_loss_landscape"
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {pdf_path} and {png_path}")
    return pdf_path, png_path


def main() -> None:
    args = parse_args()
    args.tasks = [task_name(task) for task in args.tasks]
    args.checkpoint_dir = args.checkpoint_dir.expanduser()
    args.output_dir = args.output_dir.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.save = str(args.checkpoint_dir)
    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"

    if args.plot_only:
        diagnostics_path = write_diagnostics(args)
        plot_results(args)
        if diagnostics_path:
            print(f"Saved diagnostics: {diagnostics_path}")
        return

    pending = [
        value
        for value in args.lambdas
        if args.overwrite or not result_path(args, value).is_file()
    ]
    if not pending:
        print("All requested landscape caches already exist.")
        write_diagnostics(args)
        plot_results(args)
        return

    pretrained_path = resolve_checkpoint(args, args.pretrained_pattern, args.tasks[0], 0.0)
    image_encoder = load_encoder(pretrained_path)
    names, model_params_cpu = trainable_parameters(image_encoder)
    base_params_cpu = [parameter.detach().clone() for parameter in model_params_cpu]
    verify_common_pretrained(args, names, base_params_cpu)

    image_encoder = image_encoder.to(device).eval()
    _, model_params = trainable_parameters(image_encoder, names)
    base_params = [value.to(device) for value in base_params_cpu]
    heads, batches, subset_indices = prepare_task_data(args, image_encoder)

    batches_per_point = sum(len(task_batches) for task_batches in batches)
    total_batches = len(pending) * (
        args.grid_size**2 * batches_per_point
        + args.barrier_points * batches_per_point
    )
    print(
        f"Evaluating {len(pending)} lambda value(s): approximately {total_batches} "
        "image-encoder forward batches."
    )

    ratio_by_lambda = dict(zip(args.lambdas, args.min_orthogonal_ratios))
    for coupling_lambda in pending:
        paths = specialist_paths(args, coupling_lambda)
        print(f"\nConstructing lambda={lambda_tag(coupling_lambda)} plane")
        for task, path in zip(args.tasks, paths):
            print(f"  {task}: {path}")
        u_cpu, v_cpu, diagnostics, coordinates = build_basis(
            paths, names, base_params_cpu
        )
        u = [value.detach().to(device) for value in u_cpu]
        v = [value.detach().to(device) for value in v_cpu]
        del u_cpu, v_cpu
        gc.collect()

        min_orthogonal_ratio = ratio_by_lambda[coupling_lambda]
        x_values, y_values = coordinate_grid(
            coordinates,
            args.grid_size,
            args.margin,
            min_orthogonal_ratio,
        )
        print(
            f"Grid spans: x={np.ptp(x_values):.4f}, y={np.ptp(y_values):.4f} "
            f"(minimum y/x ratio={min_orthogonal_ratio:g})."
        )
        task0_loss, task1_loss = evaluate_grid(
            image_encoder,
            model_params,
            base_params,
            u,
            v,
            x_values,
            y_values,
            heads,
            batches,
            device,
            use_amp,
            coupling_lambda,
        )
        barrier_t, barrier_losses, barriers = evaluate_barriers(
            image_encoder,
            model_params,
            base_params,
            u,
            v,
            coordinates,
            heads,
            batches,
            device,
            use_amp,
            args.barrier_points,
        )
        diagnostics.update(
            {
                "barrier_0": float(barriers[0]),
                "barrier_1": float(barriers[1]),
                "barrier_average": float(barriers.mean()),
            }
        )
        cache_path = result_path(args, coupling_lambda)
        save_result(
            cache_path,
            task_names=np.asarray(args.tasks, dtype="U"),
            x=x_values,
            y=y_values,
            task_0_loss=task0_loss,
            task_1_loss=task1_loss,
            average_loss=0.5 * (task0_loss + task1_loss),
            pretrained_coord=coordinates["pretrained"],
            specialist_0_coord=coordinates["specialist_0"],
            specialist_1_coord=coordinates["specialist_1"],
            merge_coord=coordinates["merge"],
            subset_indices_0=subset_indices[0],
            subset_indices_1=subset_indices[1],
            barrier_t=barrier_t,
            barrier_losses=barrier_losses,
            checkpoint_paths=np.asarray([str(path) for path in paths], dtype="U"),
            pretrained_checkpoint=np.asarray(str(pretrained_path), dtype="U"),
            grid_size=np.int64(args.grid_size),
            subset_size=np.int64(args.subset_size),
            seed=np.int64(args.seed),
            min_orthogonal_ratio=np.float64(min_orthogonal_ratio),
            **{"lambda": np.float64(coupling_lambda)},
            **{name: np.float64(value) for name, value in diagnostics.items()},
        )
        del u, v
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        diagnostics_path = write_diagnostics(args)
        plot_results(args)
        print(f"Updated diagnostics: {diagnostics_path}")


if __name__ == "__main__":
    main()
