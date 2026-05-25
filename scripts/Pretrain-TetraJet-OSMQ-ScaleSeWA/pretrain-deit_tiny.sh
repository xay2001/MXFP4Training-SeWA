SCRIPT_PATH=$(pwd)
DATA_PATH="/sharedspace/data/imagenet"    # Dataset Path
WORK_PATH="../../"
PATH_TO_SAVE="."                 # NEED: Path to save checkpoints

MODEL_NAME="deit_tiny"          # Model name in ["deit_tiny", "deit_small", "deit_base"]
                                # NOTE: "deit_base" needs a different Learning-Rate & Batch-Size setting
OMP_NUM_THREADS=8

EMA_DECAY=0.9983
OSMQ_START_STEP=50000
OSMQ_TAU=1.0
SCALE_SEWA_TAU=1.0
SCALE_SEWA_CURRENT_BIAS=1.0
SCALE_UPDATE_INTERVAL=8
Experiment_NAME="TetraJet-MXFP4-OSMQ-ScaleHistory-EMA${EMA_DECAY}-S${OSMQ_START_STEP}-ST${SCALE_SEWA_TAU}-SI${SCALE_UPDATE_INTERVAL}"
LOGS_NAME="logs_${Experiment_NAME}_${MODEL_NAME}"
SAVE_PATH="${PATH_TO_SAVE}/${Experiment_NAME}/${MODEL_NAME}"

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
cd "$WORK_PATH"
mkdir -p $SCRIPT_PATH/${LOGS_NAME}

# nproc_per_node: how many gpus to run on
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port=29505 main.py \
    --model ${MODEL_NAME}_patch16_224 \
    --batch-size 256 \
    --tritonQ \
    --mxscale 1 \
    --data-path $DATA_PATH \
    --output_dir $SAVE_PATH \
    --qlinear-ema-decay ${EMA_DECAY} \
    --qlinear-osmq \
    --qlinear-osmq-start-step ${OSMQ_START_STEP} \
    --qlinear-osmq-tau ${OSMQ_TAU} \
    --qlinear-scale-sewa \
    --qlinear-scale-sewa-tau ${SCALE_SEWA_TAU} \
    --qlinear-scale-sewa-init-current-bias ${SCALE_SEWA_CURRENT_BIAS} \
    --qlinear-scale-update-interval ${SCALE_UPDATE_INTERVAL} \
    --row_blocksize 1 --column_blocksize 32 \
    --qchoice all --qlinear-all \
    --fabit 4 --fwbit 4 --babit 4 --bwbit 4 \
    --faexp 2 --fwexp 2 --baexp 2 --bwexp 2 \
    > $SCRIPT_PATH/${LOGS_NAME}/${TIMESTAMP}.log 2>&1
