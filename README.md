# Zhyper: Factorized Hypernetworks for Conditioned LLM Fine-Tuning

<h1 align="center">Installation</h1>

Install `uv` if you don't have `uv` (see https://docs.astral.sh/uv/getting-started/installation/)

With `uv` installed, run the following to install the dependencies.
```bash
git clone https://github.com/SakanaAI/text-to-lora.git
cd text-to-lora
# make sure you have `uv` installed
# (see https://docs.astral.sh/uv/getting-started/installation/)
uv self update
uv venv --python 3.10 --seed
uv sync
# we use the following wheel for installation
# you might have to change the wheel to be compatible with your hardware
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
uv pip install src/fishfarm
```

---
<h2 align="center">SFT Training</h2>
For asynchronous validation evaluation, we need a separate evaluator script.
The `watcher.py` checks for new checkpoints and evaluates them as they get saved.
The script also keeps track of which one is the best checkpoint so far.

```bash
# start a watcher process for async eval
uv run watcher.py
```

Then run one of the following scripts for each GPU you have.
Each takes around 5 days on a single H100 GPU for 20k epochs and 2 days for 2k epochs.
```bash
# T2L training
./scripts/train_t2l_mistral.sh
# Zhyper training
./scripts/train_diag_zhyper_mistral.sh
./scripts/train_square_zhyper_mistral.sh
# For accelrate based training 
./scripts/start_main.sh
```
---

<h1 align="center"> Evaluation</h1>

Base model
```bash
./scripts/eval_base_models.sh
```

T2L
```bash
# example for T2L trained for gemma-2-2b-it
WANDB_MODE=disabled uv run python scripts/eval_hypermod_checkpoint.py --checkpoint_path trained_t2l/gemma_2b_t2l/hypermod.pt --full_eval --use-icl
```
`--use-icl` includes 3-shot in-context examples into evaluation queries.

