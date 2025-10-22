# Zhyper: Factorized Hypernetworks for Conditioned LLM Fine-Tuning

<h1 align="center">Installation</h1>

Install `uv` if you don't have `uv` (see https://docs.astral.sh/uv/getting-started/installation/)

With `uv` installed, run the following to install the dependencies.
```bash
git clone https://github.com/machinelearningnuremberg/Zhyper.git
cd Zhyper
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

Notes on `watcher.py`.
- `watcher.py` uses env_var `SAVE_DIR` or arg `save_dir` to read the checkpoint. This var must match the one used to save the checkpoints.
- By default this script uses a larger val set compared to T2L for a more accurate validation.
```python
Zhyper_VAL = {
        "openbookqa": {"split": "validation[:500]"},
        "hellaswag": {"split": "validation[:4000]"},
        "winogrande": {"name": "winogrande_debiased", "split": "validation[:1000]", "trust_remote_code": True},
        "boolq": {"split": "validation[:1000]"},
        "piqa": {"split": "validation[:1500]"},
        "arc_easy": {"name": "ARC-Easy", "split": "validation[:500]"},
        "arc_challenge": {"name": "ARC-Challenge", "split": "validation[:500]"},
        "gsm8k": {"name": "main", "split": "test[:500]"},
        "humaneval": {"split": "test[:30]"},
        "mbpp": {"name": "sanitized", "split": "test[:85]"}
    } # used by default (--new-eval-set True)
T2L_VAL = {
        "openbookqa": {"split": "validation[:500]"},
            "hellaswag": {"split": "train[:500]"},
            "winogrande": {"name": "winogrande_debiased", "split": "train[:500]", "trust_remote_code": True},
            "boolq": {"split": "train[:500]"},
            "piqa": {"split": "train[:500]"},
            "arc_easy": {"name": "ARC-Easy", "split": "validation[:500]"},
            "arc_challenge": {"name": "ARC-Challenge", "split": "validation[:500]"},
    }
```
- this script can be used to eval a run after or during the run is over. Simply run `SAVE_DIR="..." watcher.py`. 
`full-eval` is automtically set to True when evaluating the best run. However, can be manually set to True but assumes the best run exist outside the it_* folders of the checkpoints.

Then run one of the following scripts for each GPU you have.
Each takes around 5 days on a single H100 GPU for 20k epochs and 2 days for 2k epochs.
```bash
# T2L training
./scripts/train_t2l_mistral.sh
# Zhyper training
./scripts/train_diag_zhyper_mistral.sh
./scripts/train_square_zhyper_mistral.sh
# For accelerate-based training with async evaluation.
# Runs both training and watcher scripts with shared run_id. 
./scripts/start_main.sh
```
---

<h1 align="center"> Evaluation</h1>

Base model
```bash
./scripts/eval_base_models.sh
```

T2L/Zhyper
```bash
# example for T2L trained for gemma-2-2b-it
WANDB_MODE=disabled uv run python scripts/eval_hypermod_checkpoint.py --checkpoint_path trained_t2l/gemma_2b_t2l/hypermod.pt --full_eval --use-icl --new-eval
```
`--use-icl` includes 3-shot in-context examples into evaluation queries.

--- 

This repository was forked from T2L repository (https://github.com/SakanaAI/text-to-lora). We thank the authors for making the source code publicly available. 

