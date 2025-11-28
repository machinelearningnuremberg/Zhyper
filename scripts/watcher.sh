#!/bin/bash
#SBATCH --gres=gpu:1            # Request GPUs per node (example: 4 per node)
#SBATCH --time=1-00:00:00         # Job runtime (hh:mm:ss)
#SBATCH --partition=h100         # GPU partition/queue (depends on your cluster)
#SBATCH --output=logs/%x_%j.out    # Stdout log file: logs/jobname_jobid.out

FULL_EVAL="${1:-false}"
EXIT_ON_DONE="${2:-false}"
FULL_EVAL_WHEN_SIGNALED="${3:-true}"
NEW_EVAL_SET="${4:-true}"
FILTER_CHECKPOINTS="${5:-true}"
LOAD_STATE="${6:-true}"

# export OUTLINES_CACHE_DISABLE=1
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
# Configure Outlines cache to a node-local, per-rank absolute path to avoid
# multi-node SQLite conflicts and relative path issues.
# Prefer SLURM_TMPDIR (node-local). Fall back to /dev/shm/$USER or /tmp/$USER.
# only way to fix vllm calling in multi node
CACHE_BASE="${SLURM_TMPDIR:-/dev/shm/$USER}"
if [ ! -d "$CACHE_BASE" ]; then
    CACHE_BASE="/tmp/$USER"
fi
RANK_ID="${SLURM_LOCALID:-${LOCAL_RANK:-0}}"
NODE_ID="${SLURM_NODEID:-0}"
OUTLINES_BASE="$CACHE_BASE/outlines/$SLURM_JOB_ID/$NODE_ID"
mkdir -p "$OUTLINES_BASE/$RANK_ID"
export OUTLINES_CACHE_DIR="$OUTLINES_BASE/$RANK_ID"

source .venv/bin/activate

echo "SAVE_DIR=${SAVE_DIR} watcher.sh ${FULL_EVAL} ${EXIT_ON_DONE} ${FULL_EVAL_WHEN_SIGNALED} ${NEW_EVAL_SET} ${FILTER_CHECKPOINTS} ${LOAD_STATE}"

python watcher.py \
  --full-eval=${FULL_EVAL} \
  --exit-on-done=${EXIT_ON_DONE} \
  --full-eval-when-signaled=${FULL_EVAL_WHEN_SIGNALED} \
  --new-eval-set=${NEW_EVAL_SET} \
  --filter-checkpoints=${FILTER_CHECKPOINTS} \
  --load-state=${LOAD_STATE}


