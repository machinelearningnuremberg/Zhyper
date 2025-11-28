MODEL_NAME="${1:-mistral}"
EXP_NAME="${2:-hyper_lora}"
DATASET_TYPE="${3:-task}"
MAT_RANK="${4:-8}"
Z_TYPE="${5:-full}"
EMBED="${6:-gte}"
N_DS="${7:-479}"
LAYERS="${8:-all}"
SEED="${9:-42}"

# Set a safe default; lower it for larger ranks to avoid OOM
DS_PER_BATCH=8

if [ "$MAT_RANK" -ge 16 ]; then
    DS_PER_BATCH=4
fi

VAL_BATCH_SIZE=32
if [ "$MAT_RANK" -ge 32 ]; then
    DS_PER_BATCH=2
    VAL_BATCH_SIZE=8
fi

uuid=$(uuidgen | cut -c1-8)  # generates a random UUID
timestamp=$(date +"%Y%m%d-%H%M%S")
RUN_NAME="${timestamp}_${uuid}"
SAVE_DIR="train_outputs/sft/${EXP_NAME}/${RUN_NAME}"


export MODEL_NAME
export EXP_NAME
export DATASET_TYPE
export RUN_NAME
export SAVE_DIR
export MAT_RANK
export DS_PER_BATCH
export Z_TYPE
export EMBED
export N_DS
export LAYERS
export SEED
export VAL_BATCH_SIZE


source .venv/bin/activate
LOG_DIR_COMPUTED="${LOG_DIR:-logs}"
if [ -n "$RUN_NAME" ]; then
    SBATCH_OUTFILE="$LOG_DIR_COMPUTED/${RUN_NAME}_%x_%j.out"
else
    SBATCH_OUTFILE="$LOG_DIR_COMPUTED/%x_%j.out"
fi

export SBATCH_OUTFILE


sbatch --export=ALL --output="$SBATCH_OUTFILE" scripts/start_train.sh
