#!/usr/bin/env bash
set -euo pipefail

# Reproduce the SCouT vision lambda sweep. Override any variable on invocation,
# for example: NUM_GPUS=2 GPUS=0,1 MODEL=ViT-L-14 bash scripts/run_scout_sweep.sh

DATASETS="${DATASETS:-CIFAR100,Flowers102,PCAM,FER2013,Cars,DTD,GTSRB,RESISC45,SUN397,SVHN}"
MODEL="${MODEL:-ViT-B-32}"
RUN="${RUN:-paper_seed0}"
SAVE="${SAVE:-checkpoints/${MODEL}}"
DATA_LOCATION="${DATA_LOCATION:-${HOME}/data}"
OPENCLIP_CACHEDIR="${OPENCLIP_CACHEDIR:-${HOME}/.cache/open_clip}"
LAMBDAS="${LAMBDAS:-0 0.005 0.01 0.05 0.1 0.2 0.5 1 5 10 20}"
MERGE_MODES="${MERGE_MODES:-ta}"
COUPLING_TAU="${COUPLING_TAU:-1}"
LR="${LR:-1e-5}"
WD="${WD:-0.1}"
SEED="${SEED:-0}"
NUM_STEPS="${NUM_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_GRAD_ACCUMULATION="${NUM_GRAD_ACCUMULATION:-1}"
CLIP_MODE="${CLIP_MODE:-noclip}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU="${GPU:-0}"
GPUS="${GPUS:-0,1}"

if [[ "${NUM_GPUS}" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU}"
  TRAIN_LAUNCH=(python)
else
  export CUDA_VISIBLE_DEVICES="${GPUS}"
  TRAIN_LAUNCH=(torchrun --standalone --nproc-per-node="${NUM_GPUS}")
fi

COMMON_ARGS=(
  --model "${MODEL}"
  --run-name "${RUN}"
  --save "${SAVE}"
  --data-location "${DATA_LOCATION}"
  --openclip-cachedir "${OPENCLIP_CACHEDIR}"
  --batch-size "${BATCH_SIZE}"
)

echo "SCouT sweep: model=${MODEL}, lambdas=${LAMBDAS}, merge_modes=${MERGE_MODES}"

for coupling_lambda in ${LAMBDAS}; do
  "${TRAIN_LAUNCH[@]}" -m src.scout_finetune \
    "${COMMON_ARGS[@]}" \
    --finetuning-mode scout \
    --train-dataset "${DATASETS}" \
    --coupling-tau "${COUPLING_TAU}" \
    --coupling-lambda "${coupling_lambda}" \
    --lr "${LR}" \
    --wd "${WD}" \
    --seed "${SEED}" \
    --num-steps "${NUM_STEPS}" \
    --num-grad-accumulation "${NUM_GRAD_ACCUMULATION}" \
    --clip-mode "${CLIP_MODE}" \
    --grad-clip-norm "${GRAD_CLIP_NORM}"

  python -m src.eval_single_task \
    "${COMMON_ARGS[@]}" \
    --finetuning-mode scout \
    --eval-datasets "${DATASETS}" \
    --coupling-tau "${COUPLING_TAU}" \
    --coupling-lambda "${coupling_lambda}"

  for merge_mode in ${MERGE_MODES}; do
    python -m src.eval_task_addition \
      "${COMMON_ARGS[@]}" \
      --finetuning-mode scout \
      --eval-datasets "${DATASETS}" \
      --coupling-tau "${COUPLING_TAU}" \
      --coupling-lambda "${coupling_lambda}" \
      --merge-mode "${merge_mode}" \
      --max-coef 2 \
      --n-eval-points 41
  done
done

echo "SCouT sweep complete. Results are under ${SAVE}."
