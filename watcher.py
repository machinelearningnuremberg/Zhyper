# handmade file watcher using glob
# not using watchdog because there are too many saved files
# but we want to just watch when these files are created
# */checkpoints/it_*/hypermod.pt (for HyperLoRA) or
# */checkpoints/it_*/adapter_model.safetensors (for multi-task lora)
import itertools
import shutil
import time
import os
import argparse
from glob import glob

import numpy as np
import pandas as pd
import wandb
import yaml

from hyper_llm_modulator.sft_trainer import eval_hypermod_checkpoint, eval_lora
from hyper_llm_modulator.utils import save_yaml

HYPERLORA_CP_PATTERN = "train_outputs/sft/hyper_lora/*/checkpoints/it_*/hypermod.pt"
MTLORA_CP_PATTERN = "train_outputs/sft/mt_lora/*/checkpoints/it_*/adapter_model.safetensors"
EARLYSTOP_PATIENCE = 50


def flatten(l):
    return itertools.chain.from_iterable(l)


class Watcher:
    def __init__(self, patterns):
        self.patterns = patterns
        self.files = self.get_files()
        self.last_files = set()

    def get_files(self):
        return set(flatten(glob(pattern) for pattern in self.patterns))

    def watch(self):
        self.files = self.get_files()
        new_files = self.files - self.last_files
        self.last_files = self.files
        return new_files

    def save_state(self):
        with open("watcher_state.yaml", "w") as f:
            yaml.dump({"last_files": self.last_files}, f)

    def load_state(self):
        if not os.path.exists("watcher_state.yaml"):
            return
        with open("watcher_state.yaml", "r") as f:
            state = yaml.safe_load(f)
        if state is not None:
            if state.get("last_files"):
                self.last_files = state["last_files"]


def get_sorted_checkpoints(adapter_dir):
    checkpoints = glob(f"{adapter_dir}/checkpoints/it_*")
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("it_")[1].split("/")[0]))
    return checkpoints


def get_best_checkpoint(checkpoints):  # read saved perf csv
    dfs = []
    for cp in checkpoints:
        if glob(f"{cp}/eval_results/combined_results.csv"):
            dfs.append(pd.read_csv(f"{cp}/eval_results/combined_results.csv"))
        else:
            dfs.append(None)

    # dfs = [pd.read_csv(f"{cp}/eval_results/combined_results.csv") for cp in checkpoints]
    if "hyper_lora" in checkpoints[0]:
        best_df_idx = np.argmax(
            [df[df["split"] == "eval_descs"]["benchmark_avg"].loc[0] if df is not None else 0 for df in dfs]
        )
    elif "mt_lora" in checkpoints[0]:
        best_df_idx = np.argmax([df["benchmark_avg"].loc[0] if df is not None else 0 for df in dfs])
    best_checkpoint = checkpoints[best_df_idx]
    return best_df_idx, best_checkpoint


def save_best_checkpoint(adapter_dir, best_checkpoint):
    if "hyper_lora" in best_checkpoint:
        shutil.copy(f"{best_checkpoint}/hypermod.pt", f"{adapter_dir}/hypermod.pt")
    elif "mt_lora" in best_checkpoint:
        shutil.copy(f"{best_checkpoint}/adapter_model.safetensors", f"{adapter_dir}/adapter_model.safetensors")


def check_earlystop(adapter_dir, checkpoints, best_df_idx, best_checkpoint):
    n_since_best = len(checkpoints) - best_df_idx - 1
    if n_since_best >= EARLYSTOP_PATIENCE:
        info = dict(
            best_checkpoint=best_checkpoint,
            stopped_with_patience=EARLYSTOP_PATIENCE,
            last_checkpoint=checkpoints[-1],
        )
        save_yaml(info, f"{adapter_dir}/earlystop_info.yaml")


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "hypermod_sft"
    wandb_dir = f"{os.environ['HOME']}/.wandb/logs/hypermod_sft/"
    watcher = Watcher([HYPERLORA_CP_PATTERN, MTLORA_CP_PATTERN, HYPERLORA_CP_PATTERN_MODEL])
    watcher.load_state()
    print(vars(watcher))
    print("Watching for new files...")
    while True:
        time.sleep(10)
        new_files = watcher.watch()
        for file in new_files:
            if "checkpoints" not in file:
                continue
            # workaround to prevent loading incomplete files
            time.sleep(10)
            if not os.path.exists(file):
                # cp is delete before we can read it
                continue
            adapter_dir = file.split("checkpoints/")[0]
            args = argparse.Namespace(**yaml.safe_load(open(f"{adapter_dir}/args.yaml", "r")))
            curstep = int(file.split("it_")[1].split("/")[0])
            wandb_kwargs = {
                "project": os.getenv("WANDB_PROJECT"),
                "group": args.run_name,
                "name": f"{args.run_name}-eval",
                "id": f"{args.run_name}-eval",
                "resume": "allow",
                "dir": wandb_dir,
            }
            # init wandb run
            wandb.init(**wandb_kwargs)
            # eval
            if args.exp_setup == "z_hyper_lora":
                if "hypermod.pt" in file:
                    if "adapter_model.safetensors" in os.listdir(os.path.dirname(file)):
                        eval_hypermod_checkpoint(file, os.path.dirname(file), "cuda", curstep, full_eval=False)
            else:
                if "hypermod.pt" in file:
                    eval_hypermod_checkpoint(file, None, "cuda", curstep, full_eval=False)

                elif "adapter_model.safetensors" in file:
                    lora_dir = os.path.dirname(file)
                    eval_lora(args, lora_dir, curstep, full_eval=False)

            # get the best checkpoint
            checkpoints = get_sorted_checkpoints(adapter_dir)
            best_df_idx, best_checkpoint = get_best_checkpoint(checkpoints)
            # copy best checkpoint to hypermod_dir
            save_best_checkpoint(adapter_dir, best_checkpoint)
            # check if we should early stop
            check_earlystop(adapter_dir, checkpoints, best_df_idx, best_checkpoint)
            # close wandb run
            wandb.finish()

        watcher.save_state()
