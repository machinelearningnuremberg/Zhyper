#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:4            # Request GPUs per node (example: 4 per node)
#SBATCH --time=1-00:00:00         # Job runtime (hh:mm:ss)
#SBATCH --partition=h100         # GPU partition/queue (depends on your cluster)
#SBATCH --output=logs/%x_%j.out    # Overridden by --output passed from start_main.sh


module load intelmpi
# --exclusive
echo "Running with run_name=$RUN_NAME save_dir=$SAVE_DIR model=$MODEL_NAME exp=$EXP_NAME dataset=$DATASET_TYPE r=$MAT_RANK"

export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn

#export NCCL_ALGO=ring
export NCCL_DEBUG=info
#export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,COLL

export PYTHONFAULTHANDLER=1

export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0

export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)

CACHE_BASE="${SLURM_TMPDIR:-/dev/shm/$USER}"
if [ ! -d "$CACHE_BASE" ]; then
    CACHE_BASE="/tmp/$USER"
fi
RANK_ID="${SLURM_LOCALID:-${LOCAL_RANK:-0}}"
NODE_ID="${SLURM_NODEID:-0}"
OUTLINES_BASE="$CACHE_BASE/outlines/$SLURM_JOB_ID/$NODE_ID"
mkdir -p "$OUTLINES_BASE/$RANK_ID"
export OUTLINES_CACHE_DIR="$OUTLINES_BASE/$RANK_ID"

echo go $COUNT_NODE
echo $HOSTNAMES

rm -f watcher_fifo_${RUN_NAME}
source .venv/bin/activate
mpirun -n $COUNT_NODE -perhost 1 scripts/start_acc.sh
(sleep 60 && python3 - <<'EOF' # change to full state once done.
import yaml, os

state_file = f"watcher_state_{os.environ['RUN_NAME']}.yaml"
if os.path.exists(state_file):
    with open(state_file, "r") as f:
        state = yaml.safe_load(f) or {}
else:
    state = {}
state["full_eval"] = 1
with open(state_file, "w") as f:
    yaml.dump(state, f)
EOF
)