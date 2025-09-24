MODEL_NAME="${1:-mistral}"
EXP_NAME="${2:-hyper_lora}"
DATASET_TYPE="${3:-task}"
MAT_RANK="${4:-8}"
Z_TYPE="${5:-full}"

# Set a safe default; lower it for larger ranks to avoid OOM
DS_PER_BATCH=${DS_PER_BATCH:-8}
if [ "$MAT_RANK" -gt 16 ]; then
    DS_PER_BATCH=4
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


source .venv/bin/activate
LOG_DIR_COMPUTED="${LOG_DIR:-logs}"
if [ -n "$RUN_NAME" ]; then
    SBATCH_OUTFILE="$LOG_DIR_COMPUTED/${RUN_NAME}_%x_%j.out"
else
    SBATCH_OUTFILE="$LOG_DIR_COMPUTED/%x_%j.out"
fi

sbatch --export=ALL --output="$SBATCH_OUTFILE" scripts/start_train.sh
#   --full-eval --exit-on-done --full-eval-when-signaled --new-eval-set --filter-checkpoints --load-state
if [[ "$DATASET_TYPE" != *"align"* && "$EXP_NAME" != lora_* ]]; then
    sbatch --export=ALL --output="$SBATCH_OUTFILE" scripts/watcher.sh false false true true true true
fi
