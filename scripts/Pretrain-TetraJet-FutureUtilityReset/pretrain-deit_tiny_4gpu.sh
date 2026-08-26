#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH=$(pwd)
DATA_PATH=${DATA_PATH:-/sharedspace/data/imagenet}
WORK_PATH="../../"
PATH_TO_SAVE="."

MODEL_NAME="deit_tiny"
OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
EMA_DECAY=${EMA_DECAY:-0.9983}
FUR_START_STEP=${FUR_START_STEP:-50000}
FUR_RESET_INTERVAL=${FUR_RESET_INTERVAL:-200}
FUR_CANDIDATE_RATIO=${FUR_CANDIDATE_RATIO:-0.05}
FUR_BUDGET_RATIO=${FUR_BUDGET_RATIO:-0.01}
FUR_UTILITY_DECAY=${FUR_UTILITY_DECAY:-0.9}
FUR_FLIP_DECAY=${FUR_FLIP_DECAY:-0.95}
FUR_TAU=${FUR_TAU:-0.1}
EPOCHS=${EPOCHS:-90}
MASTER_PORT=${MASTER_PORT:-29514}
CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
BATCH_SIZE=${BATCH_SIZE:-256}
RUN_TAG=${RUN_TAG:-}

Experiment_NAME="TetraJet-MXFP4-DFUR-EMA${EMA_DECAY}-S${FUR_START_STEP}-I${FUR_RESET_INTERVAL}-C${FUR_CANDIDATE_RATIO}-B${FUR_BUDGET_RATIO}${RUN_TAG}"
LOGS_NAME="logs_${Experiment_NAME}_${MODEL_NAME}"
SAVE_PATH="${PATH_TO_SAVE}/${Experiment_NAME}/${MODEL_NAME}"

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
cd "$WORK_PATH"
mkdir -p "$SCRIPT_PATH/$LOGS_NAME"

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} PYTHONUNBUFFERED=1 python -m torch.distributed.run \
    --nproc_per_node=${NPROC_PER_NODE} --master_port=${MASTER_PORT} main.py \
    --model ${MODEL_NAME}_patch16_224 \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --tritonQ \
    --mxscale 1 \
    --data-path "$DATA_PATH" \
    --output_dir "$SAVE_PATH" \
    --qlinear-ema-decay ${EMA_DECAY} \
    --qlinear-future-utility-reset \
    --qlinear-future-utility-start-step ${FUR_START_STEP} \
    --qlinear-future-reset-interval ${FUR_RESET_INTERVAL} \
    --qlinear-future-candidate-ratio ${FUR_CANDIDATE_RATIO} \
    --qlinear-future-budget-ratio ${FUR_BUDGET_RATIO} \
    --qlinear-future-utility-decay ${FUR_UTILITY_DECAY} \
    --qlinear-future-flip-decay ${FUR_FLIP_DECAY} \
    --qlinear-future-utility-tau ${FUR_TAU} \
    --row_blocksize 1 --column_blocksize 32 \
    --qchoice all --qlinear-all \
    --fabit 4 --fwbit 4 --babit 4 --bwbit 4 \
    --faexp 2 --fwexp 2 --baexp 2 --bwexp 2 \
    > "$SCRIPT_PATH/$LOGS_NAME/${TIMESTAMP}.log" 2>&1
