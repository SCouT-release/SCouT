import argparse
import os

import torch

from src.mergopt import DEFAULT_MERGOPT_ALPHAS, parse_mergopt_alphas


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser("~/data"),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT. ",
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        type=lambda x: x.split(","),
        help="Which dataset(s) to patch on.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional run label included in checkpoint and result filenames.",
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--num-grad-accumulation",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument(
        "--uw-lr",
        type=float,
        default=1e-3,
        help="Learning rate for uncertainty-weighting parameters.",
    )
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--ls", type=float, default=0.0, help="Label smoothing.")
    parser.add_argument(
        "--clip-mode",
        choices=["global", "indept", "noclip"],
        default="noclip",
        help="Gradient clipping mode. Use indept for per-task clipping.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm used when --clip-mode is not noclip.",
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=2000,
        help="Number of optimizer steps for step-based fine-tuning.",
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",  # noqa: E501
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default=os.path.expanduser("~/openclip-cachedir/open_clip"),
        help="Directory for caching models from OpenCLIP",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of processes for distributed training.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=-1,
        help="How often to checkpoint the model.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=12355,
        help="Port for distributed training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed.",
    )
    parser.add_argument(
        "--finetuning-mode",
        choices=[
            "independent_ft",
            "ftts",
            "posthoc_ftts",
            "none",
            "saft",
            "ft_attention",
            "mergopt",
            "scout",
            "hard_mtl",
            "hard_mtl_uw",
            "hard_mtl_pcgrad",
        ],
        help="Training method, using the method names from the paper.",
    )
    parser.add_argument(
        "--saft-rho",
        type=float,
        default=0.5,
        help="ASAM rho for SAFT fine-tuning.",
    )
    parser.add_argument(
        "--mergopt-mu",
        type=float,
        default=0.0,
        help="Laplace location parameter for MergOPT merge offsets.",
    )
    parser.add_argument(
        "--mergopt-b",
        type=float,
        default=0.0005,
        help="Laplace scale parameter for MergOPT merge offsets.",
    )
    parser.add_argument(
        "--mergopt-k-max",
        type=int,
        default=None,
        help=(
            "Maximum number of merged tasks sampled by MergOPT. "
            "Defaults to the number of independently fine-tuned tasks."
        ),
    )
    parser.add_argument(
        "--mergopt-alphas",
        type=parse_mergopt_alphas,
        default=DEFAULT_MERGOPT_ALPHAS,
        help="Comma-separated merge coefficients sampled by MergOPT.",
    )
    parser.add_argument(
        "--coupling-tau",
        dest="coupling_tau",
        type=int,
        default=1,
        help="Run one SCouT coupling update after this many main updates.",
    )
    parser.add_argument(
        "--coupling-lambda",
        type=float,
        default=0.1,
        help="SCouT coupling strength.",
    )
    parser.add_argument(
        "--init-from-independent-run",
        type=str,
        default=None,
        help=(
            "Optional Independent FT run name used to initialize each "
            "SCouT specialist from its own independent checkpoint."
        ),
    )
    parser.add_argument(
        "--fisher-num-samples",
        type=int,
        default=256,
        help="Maximum validation examples per task for empirical Fisher estimation.",
    )
    parser.add_argument(
        "--fisher-batch-size",
        type=int,
        default=16,
        help="Validation loader batch size for Fisher calibration.",
    )
    parser.add_argument(
        "--fisher-identity-mix",
        type=float,
        default=0.1,
        help="Identity mixture weight for normalized Fisher metrics.",
    )
    parser.add_argument(
        "--fisher-min",
        type=float,
        default=0.05,
        help="Minimum value for normalized Fisher metrics.",
    )
    parser.add_argument(
        "--fisher-max",
        type=float,
        default=20.0,
        help="Maximum value for normalized Fisher metrics.",
    )
    parser.add_argument(
        "--fisher-floor",
        type=float,
        default=1e-6,
        help="Minimum denominator for Fisher-weighted model merging.",
    )
    parser.add_argument(
        "--n-eval-points",
        type=int,
        default=20,
        help="Number of evaluation points used to find optimal coefficient in task arithmetic.",
    )
    parser.add_argument(
        "--max-coef",
        type=float,
        default=1,
        help="Maximum scaling coefficient searched during task arithmetic evaluation.",
    )
    parser.add_argument(
        "--merge-mode",
        type=str.lower,
        choices=["ta", "average", "fisher", "ties", "wudi"],
        default="ta",
        help="Post-hoc merging rule for eval_merge.py.",
    )
    parser.add_argument(
        "--ties-trim-ratio",
        type=float,
        default=0.2,
        help="Fraction of largest-magnitude task-vector entries to keep for TIES.",
    )
    parser.add_argument(
        "--wudi-steps",
        type=int,
        default=300,
        help="Number of Adam steps for WUDI-Merging.",
    )
    parser.add_argument(
        "--wudi-lr",
        type=float,
        default=1e-5,
        help="Learning rate for WUDI-Merging.",
    )
    parser.add_argument(
        "--adamerging-steps",
        type=int,
        default=500,
        help="Number of optimizer steps for supervised AdaMerging.",
    )
    parser.add_argument(
        "--adamerging-lr",
        type=float,
        default=0.001,
        help="Learning rate for supervised AdaMerging coefficients.",
    )
    parser.add_argument(
        "--adamerging-prior",
        type=float,
        default=0.15,
        help="Initial and regularization target value for AdaMerging coefficients.",
    )
    parser.add_argument(
        "--adamerging-reg",
        type=float,
        default=0.01,
        help="L2 regularization strength toward --adamerging-prior.",
    )
    parser.add_argument(
        "--adamerging-batch-size",
        type=int,
        default=None,
        help="Validation batch size for AdaMerging coefficient optimization.",
    )
    parser.add_argument(
        "--adamerging-log-every",
        type=int,
        default=25,
        help="How often to print AdaMerging optimization progress.",
    )
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args
