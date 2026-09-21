#!/usr/bin/env bash
set -euo pipefail

# Train and evaluate the vision baselines used in the paper.

DATASETS="${DATASETS:-CIFAR100,Flowers102,PCAM,FER2013,Cars,DTD,GTSRB,RESISC45,SUN397,SVHN}"
MODEL="${MODEL:-ViT-B-32}"
RUN="${RUN:-paper_seed0}"
SAVE="${SAVE:-checkpoints/${MODEL}}"
DATA_LOCATION="${DATA_LOCATION:-${HOME}/data}"
OPENCLIP_CACHEDIR="${OPENCLIP_CACHEDIR:-${HOME}/.cache/open_clip}"
METHODS="${METHODS:-standard linear attention saft mergopt hard_joint uw pcgrad}"
MERGE_MODES="${MERGE_MODES:-ta}"
LR="${LR:-1e-5}"
WD="${WD:-0.1}"
SEED="${SEED:-0}"
NUM_STEPS="${NUM_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GPU="${GPU:-0}"

export CUDA_VISIBLE_DEVICES="${GPU}"

COMMON_ARGS=(
  --model "${MODEL}"
  --run-name "${RUN}"
  --save "${SAVE}"
  --data-location "${DATA_LOCATION}"
  --openclip-cachedir "${OPENCLIP_CACHEDIR}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --wd "${WD}"
  --seed "${SEED}"
  --num-steps "${NUM_STEPS}"
)

for method in ${METHODS}; do
  case "${method}" in
    standard|linear|attention|saft|mergopt)
      python -m src.indep_finetune \
        "${COMMON_ARGS[@]}" \
        --finetuning-mode "${method}" \
        --train-dataset "${DATASETS}"

      python -m src.eval_single_task \
        "${COMMON_ARGS[@]}" \
        --finetuning-mode "${method}" \
        --eval-datasets "${DATASETS}"

      for merge_mode in ${MERGE_MODES}; do
        python -m src.eval_task_addition \
          "${COMMON_ARGS[@]}" \
          --finetuning-mode "${method}" \
          --eval-datasets "${DATASETS}" \
          --merge-mode "${merge_mode}" \
          --max-coef 2 \
          --n-eval-points 41
      done
      ;;
    hard_joint|uw|pcgrad)
      python -m src.hard_joint_finetune \
        "${COMMON_ARGS[@]}" \
        --finetuning-mode "${method}" \
        --train-dataset "${DATASETS}"

      python -m src.eval_single_task \
        "${COMMON_ARGS[@]}" \
        --finetuning-mode "${method}" \
        --eval-datasets "${DATASETS}"
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      exit 2
      ;;
  esac
done

echo "Baseline runs complete. Results are under ${SAVE}."
