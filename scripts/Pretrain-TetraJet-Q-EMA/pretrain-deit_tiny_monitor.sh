SCRIPT_PATH=$(pwd)
DATA_PATH="/sharedspace/data/imagenet"    # Dataset Path
WORK_PATH="../../"
PATH_TO_SAVE="."                 # NEED: Path to save checkpoints

MODEL_NAME="deit_tiny"          # Model name in ["deit_tiny", "deit_small", "deit_base"]
OMP_NUM_THREADS=8

EMA_DECAY=0.9983
SCALE_MONITOR_INTERVAL=500
GPUS=${GPUS:-4,5,6,7}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-29501}
Experiment_NAME="TetraJet-MXFP4-EMA${EMA_DECAY}-Monitor"
LOGS_NAME="logs_${Experiment_NAME}_${MODEL_NAME}"
SAVE_PATH="${PATH_TO_SAVE}/${Experiment_NAME}/${MODEL_NAME}"

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
cd "$WORK_PATH"
mkdir -p $SCRIPT_PATH/${LOGS_NAME}

CUDA_VISIBLE_DEVICES=${GPUS} python -u -m torch.distributed.run --nproc_per_node=${NPROC_PER_NODE} --master_port=${MASTER_PORT} main.py \
    --model ${MODEL_NAME}_patch16_224 \
    --batch-size 256 \
    --tritonQ \
    --mxscale 1 \
    --data-path $DATA_PATH \
    --output_dir $SAVE_PATH \
    --qlinear-ema-decay ${EMA_DECAY} \
    --qlinear-scale-monitor \
    --qlinear-scale-monitor-interval ${SCALE_MONITOR_INTERVAL} \
    --row_blocksize 1 --column_blocksize 32 \
    --qchoice all --qlinear-all \
    --fabit 4 --fwbit 4 --babit 4 --bwbit 4 \
    --faexp 2 --fwexp 2 --baexp 2 --bwexp 2 \
    > $SCRIPT_PATH/${LOGS_NAME}/${TIMESTAMP}.log 2>&1
