import gc
import os
import json
import logging
import random
from copy import deepcopy
from functools import partial
from glob import glob

import torch
import pandas as pd
import wandb

from hyper_llm_modulator.hyper_modulator import load_hypermod_checkpoint, save_lora
from hyper_llm_modulator.res_aggregator import aggregrate_results_and_save_to_file
from hyper_llm_modulator.utils import generate_simplex_points, get_layers, get_metadata, save_json, log_scalar
from hyper_llm_modulator.data import BENCHMARK_TASK_INFO
from hyper_llm_modulator.utils.lora_formatting import convert_qkv_gate_up_lora_to_splits_vllm
from hyper_llm_modulator.utils.model_loading import get_tokenizer
from hyper_llm_modulator.utils.preprocessing import preprocess_result
from hyper_llm_modulator.utils.utils import embed_texts
from hyper_llm_modulator.vllm_eval import eval

logger = logging.getLogger()


def eval_hypermod_checkpoint(checkpoint_path, peft_adapter_path, device, curstep, full_eval, use_icl=False):
    # peft_adapter_path in case of using z. Here we read the LORA weights and add it to the model.
    # load checkpoint
    args, hypermod, model, tokenizer, emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn = (
        load_hypermod_checkpoint(checkpoint_path, peft_adapter_path, device)
    )
    chat_template = tokenizer.chat_template
    layer_indices = torch.tensor(range(len(get_layers(model))), dtype=torch.long, device=device)
    save_dir = os.path.dirname(checkpoint_path)
    train_metadata = get_metadata(args.train_ds_names, args.use_per_task_emb)
    val_metadata = get_metadata(args.eval_ds_info, args.use_per_task_emb)

    eval_ds_info = deepcopy(args.eval_ds_info)
    if not full_eval:
        eval_ds_info = {k: v for k, v in eval_ds_info.items() if k in BENCHMARK_TASK_INFO}
        for k in BENCHMARK_TASK_INFO:
            eval_ds_info[k]["ds_kwargs"] = BENCHMARK_TASK_INFO[k]
    
    for ds in list(eval_ds_info.keys()):
        if ds.startswith("lol_"):
            eval_ds_info.pop(ds)

    if args.use_one_hot_task_emb:
        all_lora_dirs, save_dicts = generate_lora_for_tasks_one_hot(args, 
                                                                    hypermod, 
                                                                    layer_indices, 
                                                                    save_dir, 
                                                                    device, 
                                                                    model if peft_adapter_path is not None else None)
    else:
        all_lora_dirs, save_dicts = generate_loras_for_tasks_from_descs(
            args.model_dir,
            eval_ds_info,
            args.additional_eval_descs,
            train_metadata,
            val_metadata,
            save_dir,
            device,
            layer_indices,
            emb_model,
            emb_tokenizer,
            task_desc_format_fn,
            pooling_fn,
            hypermod,
            model if peft_adapter_path is not None else None
        )
    # run vllm eval on generated loras
    del hypermod, model, tokenizer, emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn, layer_indices
    gc.collect()
    torch.cuda.empty_cache()
    for eval_ds in eval_ds_info:
        ds_kwargs = None
        if "ds_kwargs" in eval_ds_info[eval_ds]:
            ds_kwargs = eval_ds_info[eval_ds]["ds_kwargs"] if eval_ds_info[eval_ds]["ds_kwargs"] else None
        results = do_eval_task(
            args.model_dir,
            chat_template,
            save_dir,
            all_lora_dirs[eval_ds],
            eval_ds,
            save_dicts[eval_ds],
            ds_kwargs,
            use_icl,
        )
        print(results)
    # aggregate eval results
    df = aggregrate_results_and_save_to_file(
        base_model_dir=args.model_dir,
        mt_lora_dir=args.mt_lora_path,
        hypermod_dir=save_dir,
        hypermod_name="hyperlora",
    )
    if full_eval:
        out = {
            "test/benchmark/acc/other_train_descs": df["benchmark_avg"].loc[("hyperlora", "other_train_descs")],
            "test/benchmark/acc/random_descs": df["benchmark_avg"].loc[("hyperlora", "random_descs")],
        }
        if ("hyperlora", "eval_descs") in df["benchmark_avg"].index:
            out["test/benchmark/acc/eval_descs"] = df["benchmark_avg"].loc[("hyperlora", "eval_descs")]
        else:
            out["test/benchmark/acc/train_descs"] = df["benchmark_avg"].loc[("hyperlora", "train_descs")]
    else:
        out = {
            "val/benchmark/acc/other_train_descs": df["benchmark_avg"].loc[("hyperlora", "other_train_descs")],
            "val/benchmark/acc/random_descs": df["benchmark_avg"].loc[("hyperlora", "random_descs")],
        }
        if ("hyperlora", "eval_descs") in df["benchmark_avg"].index:
            out["val/benchmark/acc/eval_descs"] = df["benchmark_avg"].loc[("hyperlora", "eval_descs")]
        else:
            out["val/benchmark/acc/train_descs"] = df["benchmark_avg"].loc[("hyperlora", "train_descs")]
    if wandb.run is not None:
        wandb.log(out, step=curstep)
    return out


@torch.no_grad()
def generate_lora_for_tasks_one_hot(
    args,
    hypermod,
    layer_indices,
    save_dir,
    device,
    model
):
    splits = ["train_descs", "other_train_descs", "random_descs"]
    all_lora_dirs = {eval_task: [] for eval_task in args.eval_ds_info}
    save_dicts = {eval_task: [] for eval_task in args.eval_ds_info}
    eye = torch.eye(len(args.train_ds_names)).to(device)
    mix_emb = generate_simplex_points(n_points=3, dimension=len(args.train_ds_names)).to(device)

    for eval_task, eval_info in args.eval_ds_info.items():
        # only eval seen tasks with one-hot emb
        if eval_task not in args.train_ds_names:
            continue
        train_ds = eval_task

        logger.info("=" * 80 + f"\nGenerating LoRA for {eval_task}")
        train_idx = args.train_ds_names.index(train_ds)
        train_emb = eye[train_idx].unsqueeze(0)
        non_train_idx = [i for i in range(len(args.train_ds_names)) if i != train_idx]
        random_train_emb = None
        if non_train_idx:
            random_train_idx = random.choice(non_train_idx)
            random_train_emb = eye[random_train_idx].unsqueeze(0)
        for split, embs in zip(splits, [train_emb, random_train_emb, mix_emb]):
            if embs is not None:
                logger.info(f"{split}: {embs}")
                dirs = [f"{save_dir}/generated_loras/{eval_task}/{split}/lora_{i}" for i in range(len(embs))]
                for lora_dir, task_emb in zip(dirs, embs):
                    logger.info("=" * 80 + f"\nGenerating LoRA with task_emb = {task_emb}")
                    save_dicts[eval_task].append(
                        {"task_emb": task_emb.cpu().numpy().tolist(), "split": split, "lora_dir": lora_dir}
                    )
                    all_lora_dirs[eval_task].append(lora_dir)
                    task_emb = task_emb.unsqueeze(0)
                    encoded_task_emb = hypermod.task_encoder(task_emb)["encoded_task_emb"].detach()

                    lora_sd = hypermod.gen_lora(layer_indices, encoded_task_emb, model)
                    save_lora(lora_sd, hypermod.peft_config, lora_dir)
                    # if "Phi-3" in args.model_dir:
                    #     convert_qkv_gate_up_lora_to_splits_vllm(lora_dir)
    return all_lora_dirs, save_dicts


@torch.no_grad()
def generate_loras_for_tasks_from_descs(
    model_dir,
    eval_ds_info,
    random_descs,
    train_metadata,
    val_metadata,
    save_dir,
    device,
    layer_indices,
    emb_model,
    emb_tokenizer,
    task_desc_format_fn,
    pooling_fn,
    hypermod,
    model
):
    _gen_and_save_lora = partial(
        gen_and_save_lora,
        model_dir=model_dir,
        device=device,
        layer_indices=layer_indices,
        emb_model=emb_model,
        emb_tokenizer=emb_tokenizer,
        task_desc_format_fn=task_desc_format_fn,
        pooling_fn=pooling_fn,
        hypermod=hypermod,
        model=model
    )

    ds_descs = {ds: train_metadata[ds]["descriptions"] for ds in train_metadata}
    ds_descs.update({ds: val_metadata[ds]["descriptions"] for ds in val_metadata})

    splits = ["train_descs", "eval_descs", "other_train_descs", "random_descs"]
    all_lora_dirs = {eval_task: [] for eval_task in eval_ds_info}
    save_dicts = {eval_task: [] for eval_task in eval_ds_info}
    for eval_task, eval_info in eval_ds_info.items():
        logger.debug("=" * 80 + f"\nGenerating LoRA for {eval_task}")
        eval_descs = eval_info["descriptions"]
        # take 1 training description if the task is in the training set
        train_descs = ds_descs[eval_task][0:1] if eval_task in train_metadata else []
        # take 3 descriptions from other training datasets
        other_train_descs = [random.choice(ds_descs[ds]) for ds in ds_descs if ds != eval_task]
        other_train_descs = random.sample(other_train_descs, k=min(len(other_train_descs), 3))
        for split, descs in zip(splits, [train_descs, eval_descs, other_train_descs, random_descs]):
            logger.debug(f"{split}: {descs}")
            dirs = [f"{save_dir}/generated_loras/{eval_task}/{split}/lora_{i}" for i in range(len(descs))]
            # lora_dirs[split] = dirs
            for lora_dir, task_desc in zip(dirs, descs):
                logger.debug("=" * 80 + f"\nGenerating LoRA with task_desc = {task_desc}")
                save_dicts[eval_task].append({"task_desc": task_desc, "split": split, "lora_dir": lora_dir})
                all_lora_dirs[eval_task].append(lora_dir)
                _gen_and_save_lora(lora_dir=lora_dir, task_desc=task_desc)
    return all_lora_dirs, save_dicts


@torch.no_grad()
def gen_and_save_lora(
    model_dir,
    device,
    layer_indices,
    emb_model,
    emb_tokenizer,
    task_desc_format_fn,
    pooling_fn,
    hypermod,
    lora_dir,
    task_desc,
    model
):
    task_emb = embed_texts([task_desc], emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn, device)
    encoder_out = hypermod.task_encoder(task_emb)
    encoded_task_emb = encoder_out["encoded_task_emb"].detach()
    lora_sd = hypermod.gen_lora(layer_indices, encoded_task_emb, model)
    save_lora(lora_sd, hypermod.peft_config, lora_dir)
    hypermod.model_config.save_pretrained(lora_dir)
    if "Phi-3" in model_dir:
        convert_qkv_gate_up_lora_to_splits_vllm(lora_dir)


def eval_lora(args, lora_dir, curstep, full_eval=False, use_icl=False):
    save_dicts = None
    all_lora_dirs = [lora_dir]
    eval_ds_info = deepcopy(args.eval_ds_info)
    chat_template = get_tokenizer(args.model_dir).chat_template

    if not full_eval:
        eval_ds_info = {k: v for k, v in eval_ds_info.items() if k in BENCHMARK_TASK_INFO}
        for k in BENCHMARK_TASK_INFO:
            eval_ds_info[k]["ds_kwargs"] = BENCHMARK_TASK_INFO[k]

    for eval_ds in eval_ds_info:
        ds_kwargs = eval_ds_info[eval_ds]["ds_kwargs"] if "ds_kwargs" in eval_ds_info[eval_ds] else None
        do_eval_task(
            args.model_dir,
            chat_template,
            lora_dir,
            all_lora_dirs,
            eval_ds,
            save_dicts,
            ds_kwargs,
            use_icl,
        )

    perf_files = glob(f"{lora_dir}/eval_results/*_eval_results.json")
    perf_files = [f for f in perf_files if not f.startswith("lol")]
    avg_perf = 0
    for perf_file in perf_files:
        with open(perf_file, "r") as f:
            perf_dict = json.load(f)
        avg_perf += perf_dict[list(perf_dict.keys())[0]][0]["results"]["acc"] / len(perf_files)
    df = pd.DataFrame.from_dict(dict(model_name=["mt_lora"], split=["test"], benchmark_avg=[avg_perf]))
    df.to_csv(f"{lora_dir}/eval_results/combined_results.csv", index=False)
    if full_eval:
        log_scalar(f"test/benchmark/acc/avg", avg_perf, curstep)
    else:
        log_scalar(f"val/benchmark/acc/avg", avg_perf, curstep)
    return avg_perf


@torch.no_grad()
def do_eval_task(
    model_dir: str,
    chat_template: str | None,
    save_dir: str,
    lora_dirs: list[str],
    eval_dataset: str,
    save_dicts: list[dict] = None,
    ds_kwargs: dict = None,
    use_icl: bool = False,
):
    perf_keys = ["acc", "mbpp_base_pass@1", "humaneval_base_pass@1", "rouge1_fmeasure", "rougeL_fmeasure"]
    os.makedirs(f"{save_dir}/eval_results", exist_ok=True)
    results = {eval_dataset: []}
    if save_dicts is None:
        save_dicts = [dict() for _ in lora_dirs]
    full_results = eval(
        model_dir,
        lora_dirs,
        eval_dataset,
        chat_template,
        gpu_memory_utilization=0.6,
        ds_kwargs=ds_kwargs,
        use_icl=use_icl,
    )

    for (lora_dir, res), save_dict in zip(full_results.items(), save_dicts):
        sampled_res_details = res.sample_details[:10]
        results[eval_dataset].append(
            dict(
                results=preprocess_result(res, perf_keys),
                sampled_res_details=sampled_res_details,
                **save_dict,
            )
        )

    torch.cuda.empty_cache()
    gc.collect()
    result_path = f"{save_dir}/eval_results/{eval_dataset}_eval_results.json"
    save_json(results, result_path)
    return results
