from functools import partial
import os
from glob import glob

import yaml
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

SAVE_DIR_RECON = "train_outputs/recon/hyper_lora"
SAVE_DIR_SFT = "train_outputs/sft/hyper_lora"
COLOR_MAP = dict(base="black", lora="red", multitask_lora="blue")


def get_tasks(model_dir):
    result_files = glob(f"{model_dir}/eval_results/*_eval_results.json")
    result_files = [os.path.basename(f).split("_eval_results.json")[0] for f in result_files]
    if "checkpoint" in model_dir:
        model_dir = model_dir.split("checkpoint")[0]
    args = yaml.safe_load(open(f"{model_dir}/args.yaml"))
    train_ds_names = args["train_ds_names"]
    eval_ds_names = list(args["eval_ds_info"].keys())

    acc_train_tasks = [
        task
        for task in eval_ds_names
        if not task.startswith("lol") and task in train_ds_names and task in result_files
    ]
    rouge_train_tasks = [
        task for task in eval_ds_names if task.startswith("lol") and task in train_ds_names and task in result_files
    ]
    acc_eval_tasks = [
        task
        for task in eval_ds_names
        if not task.startswith("lol") and task not in train_ds_names and task in result_files
    ]
    rouge_eval_tasks = [
        task
        for task in eval_ds_names
        if task.startswith("lol") and task not in train_ds_names and task in result_files
    ]
    print(f"Acc train tasks: {acc_train_tasks}")
    print(f"Rouge train tasks: {rouge_train_tasks}")
    print(f"Acc eval tasks: {acc_eval_tasks}")
    print(f"Rouge eval tasks: {rouge_eval_tasks}")
    tasks = dict(
        train=dict(acc=acc_train_tasks, rouge=rouge_train_tasks), eval=dict(acc=acc_eval_tasks, rouge=rouge_eval_tasks)
    )
    return tasks


def get_ref_perf(base_model_dir, eval_ds, metric):
    res_dir = f"eval_results/{base_model_dir}/base_model/"

    base_ref_perf = None
    if os.path.exists(f"{res_dir}/{eval_ds}_eval_results.json"):
        try:
            base_ref_perf = load_to_df(f"{res_dir}/{eval_ds}_eval_results.json", eval_ds)
            base_ref_perf["split"] = base_ref_perf["model_name"] = "base"
            base_ref_perf.set_index(["model_name", "split"], inplace=True)
            base_ref_perf = base_ref_perf[metric]
        except:
            pass

    lora_ref_perf = None
    if os.path.exists(f"{res_dir}/{eval_ds}_eval_results_lora.json"):
        try:
            lora_ref_perf = load_to_df(f"{res_dir}/{eval_ds}_eval_results_lora.json", eval_ds)
            lora_ref_perf["split"] = lora_ref_perf["model_name"] = "task_specific_lora"
            lora_ref_perf.set_index(["model_name", "split"], inplace=True)
            lora_ref_perf = lora_ref_perf[metric]
        except:
            pass

    if (base_ref_perf is not None) or (lora_ref_perf is not None):
        return pd.concat([base_ref_perf, lora_ref_perf])
    else:
        return None


def get_mt_lora_perf(mt_lora_dir, eval_ds, metric):
    if mt_lora_dir is None:
        return None
    try:
        mt_lora_ref_perf = pd.json_normalize(
            json.load(open(f"{mt_lora_dir}/eval_results/{eval_ds}_eval_results.json"))[eval_ds]
        )
        mt_lora_ref_perf["split"] = mt_lora_ref_perf["model_name"] = "multitask_lora"
        mt_lora_ref_perf.set_index(["model_name", "split"], inplace=True)
        mt_lora_ref_perf = mt_lora_ref_perf[metric]
        return mt_lora_ref_perf
    except:
        return None


def load_to_df(path, task):
    eval_results = json.load(open(path))[task]
    return pd.json_normalize(eval_results)


def get_eval_results(hypermod_dir, hypermod_name, base_model_dir, mt_lora_dir, tasks, metric):
    if len(tasks) == 0:
        return None

    dfs = []
    out_tasks = []
    for task in tasks:
        perf_file = f"{hypermod_dir}/eval_results/{task}_eval_results.json"
        df = load_to_df(perf_file, task)
        if metric not in df.columns:
            continue
        print("=" * 60)
        print(f"Task: {task}")
        print("=" * 60)
        out_tasks.append(task)
        df["model_name"] = hypermod_name if hypermod_name else "hyper_lora"
        df.rename(columns={metric: f"{task}.{metric}"}, inplace=True)
        # Ensure a split column exists (non hyper evals may not include it)
        if "split" not in df.columns:
            df["split"] = "eval_descs"
        df = df.groupby(["model_name", "split"])[f"{task}.{metric}"].mean()

        ref_perf = get_ref_perf(base_model_dir, task, metric)
        if ref_perf is not None:
            ref_perf.rename(f"{task}.{metric}", inplace=True)
        mt_lora_ref_perf = None
        if mt_lora_dir:
            mt_lora_ref_perf = get_mt_lora_perf(mt_lora_dir, task, metric)
            if mt_lora_ref_perf is not None:
                mt_lora_ref_perf.rename(f"{task}.{metric}", inplace=True)

        if ref_perf is not None or mt_lora_ref_perf is not None:
            ref_perf = pd.concat([ref_perf, mt_lora_ref_perf])
        dfs.append(pd.concat([df, ref_perf]))
        print(dfs[-1])
    if len(dfs) == 0:
        return None
    out = combined_df = pd.concat(dfs, axis=1)

    if len(out_tasks) > 1:
        avg_perf_over_tasks = combined_df.mean(axis=1)

        print("\nAverage Performance Over All Tasks:")
        print(avg_perf_over_tasks)
        out = pd.concat([combined_df, avg_perf_over_tasks], axis=1)
        out.columns = [f"{t}.{metric}" for t in out_tasks] + [f"avg.{metric}"]
    # out.to_csv(f"{hypermod_dir}/eval_results/combined_results.csv")
    return out


def aggregrate_results_and_save_to_file(hypermod_dir, hypermod_name, base_model_dir, mt_lora_dir):
    tasks = get_tasks(hypermod_dir)
    dfs = []
    _get_eval_results = partial(
        get_eval_results,
        hypermod_dir=hypermod_dir,
        hypermod_name=hypermod_name,
        base_model_dir=base_model_dir,
        mt_lora_dir=mt_lora_dir,
    )

    for task_list in [tasks["train"]["rouge"], tasks["eval"]["rouge"]]:
        _df = _get_eval_results(tasks=task_list, metric="results.rougeL_fmeasure")
        if _df is not None:
            dfs.append(_df)

    for task_list in [tasks["train"]["acc"], tasks["eval"]["acc"]]:
        _df = _get_eval_results(tasks=task_list, metric="results.acc")
        if _df is not None:
            dfs.append(_df)

    if "humaneval" in tasks["eval"]["acc"] or "humaneval" in tasks["train"]["acc"]:
        _df = _get_eval_results(tasks=["humaneval"], metric="results.humaneval_base_pass@1")
        if _df is not None:
            dfs.append(_df)
    if "mbpp" in tasks["eval"]["acc"] or "mbpp" in tasks["train"]["acc"]:
        _df = _get_eval_results(tasks=["mbpp"], metric="results.mbpp_base_pass@1")
        if _df is not None:
            dfs.append(_df)

    combined_df = pd.concat(dfs, axis=1)
    benchmark_task_cols = [
        col for col in combined_df.columns if ("acc" in col or "human" in col or "mbpp" in col) and not "avg" in col
    ]
    combined_df["benchmark_avg"] = combined_df[benchmark_task_cols].mean(axis=1)
    combined_df.to_csv(f"{hypermod_dir}/eval_results/combined_results.csv", float_format="%.5f")
    print(f"Saved to {hypermod_dir}/eval_results/combined_results.csv")
    print(combined_df["benchmark_avg"])
    return combined_df
