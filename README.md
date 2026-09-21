# SCouT: Softly Coupled Fine-Tuning

Anonymous reference implementation for the vision experiments in the ICLR 2027
submission *Softly Coupled Fine-Tuning to Preserve Specialization while
Improving Mergeability*.

SCouT trains one specialist per task while coupling the specialists to their
current merged model. The coupling strength controls the trade-off between
independent fine-tuning and hard parameter-sharing multitask learning.

> **Scope.** This public release covers the CLIP image-classification
> experiments, vision baselines, post-hoc merging methods, and training-cost
> benchmark. The RoBERTa/GLUE experiments described in the paper are not part
> of this code release.

## Included methods

| Category | Methods |
| --- | --- |
| Proposed method | SCouT |
| Specialist baselines | Independent FT, FTTS, FT-Attention, SAFT, MergOPT |
| Hard-sharing baselines | Hard MTL, uncertainty weighting, PCGrad |
| Post-hoc merging | Weight averaging, Task Arithmetic, Fisher, TIES, WUDI, AdaMerging |

The public CLI uses the name `scout`. Internally, the legacy name `soft_joint`
is retained so existing checkpoints remain compatible.

## Installation

The experiments were developed with Python 3.10, PyTorch 1.13.1, CUDA 11.6,
and OpenCLIP 2.10.1.

```bash
conda env create -f environment.yml
conda activate scout
```

Run commands from the repository root. No `PYTHONPATH` modification is needed.

## Data and pretrained weights

The vision benchmark contains CIFAR-100, Flowers102, PCAM, FER2013, Cars, DTD,
GTSRB, RESISC45, SUN397, and SVHN. Loaders use TorchVision or Hugging Face when
the dataset is available there and cache data below `--data-location`.
OpenCLIP weights are cached below `--openclip-cachedir`.

```bash
export DATA_LOCATION="$HOME/data"
export OPENCLIP_CACHEDIR="$HOME/.cache/open_clip"
```

Dataset use remains subject to each dataset's original license. Internet access
is required for the first download.

## Quick check

The unit tests do not download models or datasets:

```bash
python -m unittest discover -s tests -v
```

The same lint, compile, and unit-test checks run automatically in
`.github/workflows/ci.yml` on every push and pull request.

A short two-task SCouT run can be used to validate the end-to-end pipeline:

```bash
python -m src.scout_finetune \
  --finetuning-mode scout \
  --train-dataset CIFAR100,Flowers102 \
  --model ViT-B-32 \
  --coupling-lambda 0.5 \
  --num-steps 10 \
  --batch-size 32 \
  --save checkpoints/smoke
```

## SCouT vision sweep

`scripts/run_scout_sweep.sh` uses the paper's vision defaults: AdamW, learning
rate `1e-5`, weight decay `0.1`, batch size `128`, 2,000 steps, and
`lambda={0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 5, 10, 20}`.

```bash
# Single GPU, Task Arithmetic evaluation
bash scripts/run_scout_sweep.sh

# Two GPUs and all post-hoc merging rules from Table 2
NUM_GPUS=2 GPUS=0,1 MERGE_MODES="average ta fisher ties wudi" \
  bash scripts/run_scout_sweep.sh
```

The scripts select merging coefficients on validation sets over `[0, 2]` and
then evaluate the selected coefficient on test sets.

## Baselines

Run all training baselines and Task Arithmetic evaluation:

```bash
bash scripts/run_vision_baselines.sh
```

Methods and merge rules can be restricted without editing the script:

```bash
METHODS="standard linear attention" MERGE_MODES="average ta ties" \
  bash scripts/run_vision_baselines.sh
```

AdaMerging is evaluated separately after SCouT checkpoints are available:

```bash
python -m src.eval_adamerge \
  --finetuning-mode scout \
  --eval-datasets CIFAR100,Flowers102,PCAM,FER2013,Cars,DTD,GTSRB,RESISC45,SUN397,SVHN \
  --model ViT-B-32 \
  --coupling-lambda 0.5 \
  --run-name paper_seed0 \
  --save checkpoints/ViT-B-32
```

## Individual entry points

```bash
# Independent specialist methods
python -m src.indep_finetune --finetuning-mode standard --train-dataset CIFAR100
python -m src.indep_finetune --finetuning-mode linear --train-dataset CIFAR100
python -m src.indep_finetune --finetuning-mode attention --train-dataset CIFAR100
python -m src.indep_finetune --finetuning-mode saft --train-dataset CIFAR100
python -m src.indep_finetune --finetuning-mode mergopt --train-dataset CIFAR100

# Shared-encoder methods
python -m src.hard_joint_finetune --finetuning-mode hard_joint --train-dataset CIFAR100,Flowers102
python -m src.hard_joint_finetune --finetuning-mode uw --train-dataset CIFAR100,Flowers102
python -m src.hard_joint_finetune --finetuning-mode pcgrad --train-dataset CIFAR100,Flowers102

# Specialist and merged-model evaluation
python -m src.eval_single_task --finetuning-mode scout --eval-datasets CIFAR100,Flowers102
python -m src.eval_task_addition --finetuning-mode scout --merge-mode ta --eval-datasets CIFAR100,Flowers102
```

## Paper diagnostics

The two retained diagnostic scripts correspond to reported vision experiments:

```bash
# Loss landscape in Figure 4 and the appendix
python scripts/landscape_sweep.py --help

# Runtime and peak-memory results in Tables 8-9
torchrun --standalone --nproc-per-node=2 scripts/benchmark_training_cost.py --help
```

The resident-specialist variant is available as
`scripts/benchmark_training_cost_resident.py`.

## Outputs

Checkpoints and JSON metrics are written under `--save` and are ignored by Git.
Use a distinct `--run-name` for every seed or hyperparameter setting. The
repository intentionally does not include pretrained weights, generated plots,
or large intermediate results.

## Reproducibility notes

- Validation splits select coupling strengths and merging coefficients; test
  splits are evaluated only after selection.
- Set `--seed` explicitly for every run and report all seeds used.
- SCouT supports task sharding through `torchrun`; each process must receive at
  least one task.
- The paper's primary vision results used two NVIDIA L40S GPUs with 45 GB each.
- Exact commands print their resolved model, tasks, optimization settings, and
  output location before training.

## License and attribution

The code is released under the MIT License. This repository derives from the
MIT-licensed Task Arithmetic in the Tangent Space implementation; required
notices and upstream links are provided in `LICENSE` and
`THIRD_PARTY_NOTICES.md`.
