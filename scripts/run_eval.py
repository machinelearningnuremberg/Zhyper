import argparse
import logging
import gc
import os

import torch

from hyper_llm_modulator.utils import save_json, get_tokenizer
from hyper_llm_modulator.vllm_eval import eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--lora-dirs", nargs="+", default=None)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--use-icl", action="store_true")
    parser.add_argument("--use-task-desc", action="store_true")
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--save-to-base-model-dir", action="store_true")
    args = parser.parse_args()
    print(args)
    if isinstance(args.lora_dirs, str):
        args.lora_dirs = [args.lora_dirs]
    tokenizer = get_tokenizer(args.model_dir)
    for task in args.tasks:
        print(f"Evaluating {task}")
        json_name = f"{task}_eval_results"
        if args.use_icl:
            json_name += "_icl"
        if args.use_task_desc:
            json_name += "_task_desc"
        res = eval(
            args.model_dir,
            args.lora_dirs,
            task=task,
            chat_template=tokenizer.chat_template,
            use_icl=args.use_icl,
            use_task_desc=args.use_task_desc,
        )

        for k in res:
            print(k)
            print(res[k].aggregate_metrics)
            if args.save_results and args.lora_dirs:
                result_path = f"{k}/eval_results/{json_name}.json"
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                correctness_values = []
                for sample in res[k].sample_details:
                    sample_correctness = {}
                    if 'base_correct' in sample:
                        sample_correctness["base_correct"] = sample['base_correct']
                    if 'plus_correct' in sample:
                        sample_correctness["plus_correct"] = sample['plus_correct']
                    if 'is_correct' in sample:
                        sample_correctness["is_correct"] = sample['is_correct']
                    if 'correct' in sample:
                        sample_correctness["correct"] = sample['correct']
                    correctness_values.append(sample_correctness)
                save_json(
                    {
                        task: [
                            dict(
                                results=res[k].aggregate_metrics,
                                sampled_res_details=res[k].sample_details[:10],
                                sample_correctness=correctness_values,
                                path=k,
                            )
                        ]
                    },
                    result_path,
                )

        if args.save_to_base_model_dir:
            if not args.lora_dirs:
                result_path = (
                    f"{args.model_dir}/eval_results/base_model/{json_name}.json"
                )
            else:
                result_path = (
                    f"{args.lora_dirs}/eval_results/lora/{json_name}_lora.json"
                )
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            save_json(
                {task: [dict(results=res[k].aggregate_metrics, path=k) for k in res]},
                result_path,
            )
        torch.cuda.empty_cache()
        gc.collect()
