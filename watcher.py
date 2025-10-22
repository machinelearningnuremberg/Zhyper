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
import sys
import logging
from glob import glob

# Add src directory to Python path to find hyper_llm_modulator

import numpy as np
import pandas as pd
import wandb
import yaml

# Lazy import to avoid GPU initialization during startup
from hyper_llm_modulator.utils.eval_hypermod import eval_hypermod_checkpoint, eval_lora
from hyper_llm_modulator.utils import save_yaml
import os
from distutils.util import strtobool

SETTING = "z_hyper_vera"
HYPERLORA_CP_PATTERN = f"train_outputs/sft/{SETTING}/*/checkpoints/it_*/hypermod.pt"
HYPERLORA_CP_PATTERN_MODEL = f"train_outputs/sft/{SETTING}/*/checkpoints/it_*/adapter_model.safetensors"
MTLORA_CP_PATTERN = "train_outputs/sft/mt_lora/*/checkpoints/it_*/adapter_model.safetensors"
EARLYSTOP_PATIENCE = 50


def flatten(l):
    return itertools.chain.from_iterable(l)


class Watcher:
    def __init__(self, patterns, run_name=""):
        self.patterns = patterns
        self.files = self.get_files()
        self.last_files = self.files
        self.state_path = f"watcher_state_{run_name}.yaml"

    def get_files(self):
        return set(flatten(glob(pattern) for pattern in self.patterns))

    def watch(self):
        self.files = self.get_files()
        new_files = self.files - self.last_files
        self.last_files = self.files
        return new_files

    def save_state(self):
        state_file = self.state_path
        new_state = {}
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                state = yaml.safe_load(f) or {}
            if "full_eval" in state:
                new_state["full_eval"] = state["full_eval"]
        new_state["last_files"] = self.last_files
        # Save back
        with open(state_file, "w") as f:
            yaml.dump(new_state, f)

    def load_state(self):
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r") as f:
                state = yaml.safe_load(f)
            if state and state.get("last_files"):
                self.last_files = set(state["last_files"])
        except yaml.YAMLError as e:
            logging.warning(f"Could not load watcher_state.yaml ({e}), starting fresh.")
            self.last_files = set()


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
    if "hyper" in checkpoints[0]:
        best_df_idx = np.argmax(
            [df[df["split"] == "eval_descs"]["benchmark_avg"].loc[0] if df is not None else 0 for df in dfs]
        )
    elif ("mt_lora" in checkpoints[0]) or ("mt_vera" in checkpoints[0]):
        best_df_idx = np.argmax([df["benchmark_avg"].loc[0] if df is not None else 0 for df in dfs])
    best_checkpoint = checkpoints[best_df_idx]
    return best_df_idx, best_checkpoint


def save_best_checkpoint(adapter_dir, best_checkpoint):
    if "hyper" in best_checkpoint:
        shutil.copy(f"{best_checkpoint}/hypermod.pt", f"{adapter_dir}/hypermod.pt")
    if ("z" in best_checkpoint) or ("mt_lora" in best_checkpoint) or ("mt_vera" in best_checkpoint):
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

def to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int,)):
        return x != 0
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(x)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger()
    
    logger.info("starting watcher...")
    parser = argparse.ArgumentParser(description="Watch training outputs and trigger evals.")
    parser.add_argument("--full-eval", default=False, help="Evaluate outside checkpoints.")
    parser.add_argument("--exit-on-done", default=False, help="Exit after processing current files.")
    parser.add_argument("--full-eval-when-signaled", default=True, help="Switch to full eval when state file requests it.")
    parser.add_argument("--new-eval-set", default=True, help="Use the new extended eval set.")
    parser.add_argument("--filter-checkpoints", default=True, help="Pick best checkpoint and early stop when applicable.")
    parser.add_argument("--load-state", default=True, help="Load previous watcher state on startup.")

    parser.add_argument("--save-dir", type=str, default=os.environ.get("SAVE_DIR", ""), help="Override base run directory; defaults to $SAVE_DIR if set.")
    cli = parser.parse_args()

    full_eval = to_bool(cli.full_eval)
    exit_on_done = to_bool(cli.exit_on_done)
    full_eval_when_signaled = to_bool(cli.full_eval_when_signaled)
    new_eval_set = to_bool(cli.new_eval_set)
    filter_checkpoints = to_bool(cli.filter_checkpoints)
    load_state = to_bool(cli.load_state)
    sync_wandb = False
    wandb_id = None

    logger.info("Args: %s", vars(cli))

    save_dir = cli.save_dir
    run_name = os.path.basename(save_dir)

    if save_dir != "":
        HYPERLORA_CP_PATTERN = f"{save_dir}/checkpoints/it_*/hypermod.pt"
        HYPERLORA_CP_PATTERN_MODEL = f"{save_dir}/checkpoints/it_*/adapter_model.safetensors"
        MTLORA_CP_PATTERN = f"{save_dir}/checkpoints/it_*/adapter_model.safetensors"

    os.environ["WANDB_PROJECT"] = "hypermod_sft"
    wandb_dir = f"{os.environ['HOME']}/.wandb/logs/hypermod_sft/"
    watcher = Watcher([HYPERLORA_CP_PATTERN, MTLORA_CP_PATTERN, HYPERLORA_CP_PATTERN_MODEL], run_name=run_name)
    logger.info("watching for...")
    logger.info(f"Patterns: {[HYPERLORA_CP_PATTERN, MTLORA_CP_PATTERN, HYPERLORA_CP_PATTERN_MODEL]}")
    logger.info(f"Watcher state: {vars(watcher)}")
    if load_state:
        watcher.load_state()
    else:
        watcher.last_files = set()
    logger.info(f"Watcher state: {vars(watcher)}")
    logger.info("Watching for new files...")
    if full_eval:
        # take best run outside of checkpoints
        watcher.patterns = [pattern.replace("/checkpoints/it_*", "") for pattern in watcher.patterns]
        logger.info("Watcher is in full eval.")
    full_eval_next = 1
    full_eval_next_loop = False
    while True:
        time.sleep(10)
        # communication between main process and watcher
        # in case the watcher is used to filter the runs.
        new_files = watcher.watch()
        if len(new_files) > 0:
            new_files = sorted(new_files) # sort to avoid logging issues.
        if full_eval_when_signaled:
            if os.path.exists(watcher.state_path) and not full_eval_next_loop:
                with open(watcher.state_path, "r") as f:
                    state = yaml.safe_load(f) or {}

                if state.get("full_eval") == 1 or state.get("full_eval") is True:
                    if len(new_files) == 0: # only full eval when no more checkpoints are there and train is done.
                        full_eval_next_loop = True
                        # take best run outside of checkpoints
                        watcher.patterns = [pattern.replace("/checkpoints/it_*", "") for pattern in watcher.patterns]
                        logger.info(f"Watcher state: {vars(watcher)}")
                        logger.info("Watcher is in full eval after finishing current files.")

        for file in new_files:
            full_eval_arg = (full_eval_next == 0) or full_eval
            if ("checkpoints" not in file) and (not full_eval_arg):
                continue
            # workaround to prevent loading incomplete files
            time.sleep(10)
            if not os.path.exists(file):
                # cp is delete before we can read it
                continue
            if "checkpoints" in file:
                adapter_dir = file.split("checkpoints/")[0]
                curstep = int(file.split("it_")[1].split("/")[0])
            else:
                adapter_dir = os.path.dirname(file)
                curstep = None
            args = argparse.Namespace(**yaml.safe_load(open(f"{adapter_dir}/args.yaml", "r")))
            postfix_name = "eval" if not new_eval_set else "eval_new"
            if sync_wandb: 
                results_path = os.path.join(os.path.dirname(file), "eval_results", "combined_results.csv")
                if not os.path.exists(results_path):
                    continue # this run is not complete.
            if wandb_id is None:
                wandb_id = args.run_name + "2" if new_eval_set else "1"

            wandb_kwargs = {
                "project": os.getenv("WANDB_PROJECT"),
                "group": args.run_name,
                "name": f"{args.run_name}-{postfix_name}",
                "id": wandb_id,
                "resume": "allow",
                "dir": wandb_dir,
            }
            # init wandb run
            wandb.init(**wandb_kwargs, config=vars(args))

            if sync_wandb: 
                logging.info("Syncing wandb")
                results_path = os.path.join(os.path.dirname(file), "eval_results", "combined_results.csv")
                results_df = pd.read_csv(results_path)
                if full_eval_arg:
                    out = {}
                    # Filter for hyperlora model and check for each split
                    hyperlora_data = results_df[results_df["model_name"] == "hyperlora"]
                    
                    other_train_row = hyperlora_data[hyperlora_data["split"] == "other_train_descs"]
                    if not other_train_row.empty:
                        out["test/benchmark/acc/other_train_descs"] = other_train_row["benchmark_avg"].iloc[0]
                    
                    random_row = hyperlora_data[hyperlora_data["split"] == "random_descs"]
                    if not random_row.empty:
                        out["test/benchmark/acc/random_descs"] = random_row["benchmark_avg"].iloc[0]
                    
                    eval_row = hyperlora_data[hyperlora_data["split"] == "eval_descs"]
                    if not eval_row.empty:
                        out["test/benchmark/acc/eval_descs"] = eval_row["benchmark_avg"].iloc[0]
                    else:
                        train_row = hyperlora_data[hyperlora_data["split"] == "train_descs"]
                        if not train_row.empty:
                            out["test/benchmark/acc/train_descs"] = train_row["benchmark_avg"].iloc[0]
                else:
                    out = {}
                    # Filter for hyperlora model and check for each split
                    hyperlora_data = results_df[results_df["model_name"] == "hyperlora"]
                    
                    other_train_row = hyperlora_data[hyperlora_data["split"] == "other_train_descs"]
                    if not other_train_row.empty:
                        out["val/benchmark/acc/other_train_descs"] = other_train_row["benchmark_avg"].iloc[0]
                    
                    random_row = hyperlora_data[hyperlora_data["split"] == "random_descs"]
                    if not random_row.empty:
                        out["val/benchmark/acc/random_descs"] = random_row["benchmark_avg"].iloc[0]
                    
                    eval_row = hyperlora_data[hyperlora_data["split"] == "eval_descs"]
                    if not eval_row.empty:
                        out["val/benchmark/acc/eval_descs"] = eval_row["benchmark_avg"].iloc[0]
                    else:
                        train_row = hyperlora_data[hyperlora_data["split"] == "train_descs"]
                        if not train_row.empty:
                            out["val/benchmark/acc/train_descs"] = train_row["benchmark_avg"].iloc[0]
                # Log the results to wandb and finish the run
                if wandb.run is not None and out:
                    logging.info(f"logging wandb {out}")
                    wandb.log(out, step=curstep)
                wandb.finish()
                continue
            
            # eval
            if "z" in args.exp_setup:
                if "hypermod.pt" in file:
                    if "adapter_model.safetensors" in os.listdir(os.path.dirname(file)):
                        eval_hypermod_checkpoint(file, "cuda", curstep, full_eval=full_eval_arg, new_eval=new_eval_set)
            else:
                if "hypermod.pt" in file:
                    eval_hypermod_checkpoint(file, "cuda", curstep, full_eval=full_eval_arg, new_eval=new_eval_set)

                elif "adapter_model.safetensors" in file:
                    lora_dir = os.path.dirname(file)
                    eval_lora(args, lora_dir, curstep, full_eval=full_eval_arg, new_eval=new_eval_set)

            # get the best checkpoint
            if filter_checkpoints:
                checkpoints = get_sorted_checkpoints(adapter_dir)
                best_df_idx, best_checkpoint = get_best_checkpoint(checkpoints)
                # copy best checkpoint to hypermod_dir
                save_best_checkpoint(adapter_dir, best_checkpoint)
                # check if we should early stop
                check_earlystop(adapter_dir, checkpoints, best_df_idx, best_checkpoint)
            # close wandb run
            wandb.finish()
        watcher.save_state()
        if exit_on_done:
            break
        if full_eval_next_loop and full_eval_next == 1:
            full_eval_next-=1
            exit_on_done = True
