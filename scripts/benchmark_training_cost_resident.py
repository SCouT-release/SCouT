#!/usr/bin/env python3
"""Benchmark FT methods with resident round-robin task shards.

This is the resident-specialist counterpart of
``scripts/benchmark_training_cost.py``.  It reuses that benchmark's CLI,
method implementations, warm-up, synchronized timing, sanity checks, and
CSV/JSON output.  The only training-policy change is for the specialist
methods (Independent FT, FTTS, FT-Attention, SAFT, and MergOPT): instead of
running one task per GPU in sequential waves, rank ``r`` keeps
``datasets[r::world_size]`` resident and advances every local task once per
global step, matching SCouT's task placement and timing boundary.

Each specialist retains its own optimizer, so the underlying method semantics
are unchanged.  This policy can require substantially more VRAM, especially
for FTTS.

Example for two GPUs::

    torchrun --standalone --nproc-per-node=2 \
        scripts/benchmark_training_cost_resident.py \
        --output-dir results/training_cost_comparison_resident
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

import benchmark_training_cost as benchmark  # noqa: E402

RESIDENT_SPECIALIST_METHODS = benchmark.INDEPENDENT_METHODS
_benchmark_method_with_standard_dispatch = benchmark.benchmark_method
_write_standard_outputs = benchmark._write_outputs


def _resident_benchmark_method(method, architecture, args, datasets, device):
    """Run specialist methods through the normal round-robin shard runtime."""
    result = _benchmark_method_with_standard_dispatch(
        method, architecture, args, datasets, device
    )
    if method not in RESIDENT_SPECIALIST_METHODS:
        return result

    world_size = benchmark.get_world_size()
    task_counts = [len(datasets[rank::world_size]) for rank in range(world_size)]
    result["execution_strategy"] = (
        "resident independent specialists with round-robin task shards"
    )
    result["execution_details"] = {
        "task_assignment": "datasets[rank::world_size]",
        "resident_specialists_per_rank": task_counts,
        "max_resident_specialists_per_gpu": max(task_counts),
        "global_step_aggregation": (
            "maximum rank wall time after every local task advances once"
        ),
        "optimizer_policy": "one method-specific optimizer per specialist",
    }
    result["sanity_checks"].update(
        {
            "resident_task_specialists": True,
            "all_local_tasks_share_one_timing_boundary": True,
        }
    )
    return result


def _write_resident_outputs(payload, output_dir, prefix):
    """Correct policy metadata before delegating to the shared writer."""
    configuration = payload["configuration"]
    num_tasks = configuration["num_tasks"]
    num_gpus = configuration["num_gpus"]
    configuration.update(
        {
            "resource_policy": (
                "fixed GPU count with resident round-robin task shards for all methods"
            ),
            "task_distribution": {
                "independent_methods": (
                    "all task specialists persist in round-robin GPU shards; "
                    "each specialist retains its own optimizer"
                ),
                "SCouT": "all task specialists persist in round-robin GPU shards",
                "Hard MTL": (
                    "DDP shared-encoder replicas with round-robin task shards"
                ),
            },
            "independent_global_step_time": (
                "maximum per-rank wall time after all resident local specialists "
                "advance once"
            ),
            "peak_vram_definition": (
                "maximum allocated GiB over devices with all local task models resident"
            ),
            "resident_specialists_per_gpu": {
                "minimum": num_tasks // num_gpus,
                "maximum": math.ceil(num_tasks / num_gpus),
            },
        }
    )
    return _write_standard_outputs(payload, output_dir, prefix)


def main(argv=None):
    # The shared dispatcher sends members of INDEPENDENT_METHODS to the wave
    # benchmark.  Emptying only this dispatch set makes those methods use
    # _build_runtime(), whose round-robin sharding is already used by SCouT.
    benchmark.INDEPENDENT_METHODS = frozenset()
    benchmark.benchmark_method = _resident_benchmark_method
    benchmark._write_outputs = _write_resident_outputs
    benchmark.__doc__ = __doc__
    return benchmark.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
