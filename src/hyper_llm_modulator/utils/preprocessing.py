import random
from typing import Callable, Literal

import pandas as pd

SUM_Q = [
    "Summarize the text.",
    "Please summarize the text.",
    "Can you summarize the text?",
    "Give a summary of the text.",
    "Summarize the text for me.",
    "Please give a summary of the text.",
    "Can you give a summary of the text?",
    "Summarize.",
    "Summarize the text, please.",
    "Summarize the text, thank you.",
    "Give me a summary of the text.",
    "Please give me a summary of the text.",
]


def get_preprocessing_fn(ds_name):
    f = lambda x: x
    if ds_name.startswith("lol_"):

        def f(example):
            txt = example["input"]
            task_def = txt.split("Definition: ")[1].split("\n\nPositive Example")[0]
            task_def += " Please complete the task without any explanation."
            if len(example["output"]) > 1:
                task_def += "\nThe answer should be a comma-separated list of possible completions."
            problem = txt.split("Now complete the following example -")[1].split("Input: ")[1].split("\nOutput:")[0]
            answer = ", ".join(example["output"])
            return dict(task_def=task_def, problem=problem, answer=answer)

    if ds_name.startswith("arc_"):
        ABCD = ["A", "B", "C", "D"]

        def f(example):
            choices = example["choices"]
            assert len(choices["text"]) == len(choices["label"])
            n_to_fill = 4 - len(choices["text"])
            if len(choices["text"]) < 4:
                choices["text"] += ["N/A"] * n_to_fill
            if len(choices["label"]) < 4:
                if choices["label"][0].isdigit():
                    choices["label"] += [str(len(choices["label"]) + i + 1) for i in range(n_to_fill)]
                else:
                    choices["label"] += [ABCD[len(choices["label"]) + i] for i in range(n_to_fill)]
            example["choices"] = choices
            return example

    if ds_name.startswith("mbpp"):
        # for training an oracle lora on mbpp
        def f(example):
            example["assertions"] = "\n".join(example["test_list"])
            return example

    if "pwc" in ds_name:

        def f(example):
            return dict(context=example["input"], query=example["prompt"], answer=example["answer"])

    if "booksum" in ds_name:

        def f(example):
            context = example["chapter"].strip()
            query = random.sample(SUM_Q, 1)[0]
            return dict(context=context, query=query, answer=example["summary_text"])

    if ds_name.startswith("cul_"):
        def f(example):
            return dict(question=example["submission_title"], answer=example["comment_body"])

    return f


def add_full_stop(s):
    s = s.strip()
    # check if s ends with . or .*
    if s[-1].isalpha():
        s += "."
    return s


def preprocess_result(res, perf_keys):
    out = dict()
    agg_metrics = res.aggregate_metrics
    for k in perf_keys:
        if k in agg_metrics:
            out[k] = agg_metrics[k]
    return out


def apply_sfr_template(query: str) -> str:
    # from https://github.com/microsoft/unilm/blob/9c0f1ff7ca53431fe47d2637dfe253643d94185b/e5/utils.py#L106
    task_description = "Retrieve semantically similar text."
    return f"Instruct: {task_description}\nQuery: {query}"


def get_prompt_formatting_fn(
    metadata,
    sft_mode: Literal["causal_lm", "completion"],
    apply_chat_template_fn: Callable,
    is_intx_model: bool,
):
    assert sft_mode in ["causal_lm", "completion"], f"Invalid training task: {sft_mode}"

    def f(example):
        output_texts = dict(text=[]) if sft_mode == "causal_lm" else dict(prompt=[], response=[])
        df = pd.DataFrame(dict(example))
        for i, inp_txt in df.iterrows():
            if sft_mode == "causal_lm":
                text = metadata["text_template"].format(**inp_txt)
                output_texts["text"].append(text)
            elif sft_mode == "completion":
                prompt = metadata["user_prompt_template"].format(**inp_txt)
                output_texts["prompt"].append(prompt)
                output_texts["response"].append(str(inp_txt[metadata["response_field"]]))
        return output_texts

    def f_intx(example):
        output_texts = dict(text=[]) if sft_mode == "causal_lm" else dict(prompt=[], response=[])
        df = pd.DataFrame(dict(example))
        for i, inp_txt in df.iterrows():
            # NOTE: we assume specific chat_template here
            # that the chat_template should not have a default system_message
            # and it skils the system header if system_message is not provided
            # that is, using apply_chat_template to response_chat would not add the system_message
            prompt_chat = [
                {"role": "system", "content": metadata["system_message"].format(**inp_txt)},
                {"role": "user", "content": metadata["user_prompt_template"].format(**inp_txt)},
            ]
            response_chat = [
                {
                    "role": "assistant",
                    "content": metadata["assistant_prefill"].format(**inp_txt)
                    + str(inp_txt[metadata["response_field"]]),
                }
            ]
            if "assistant_postfill" in metadata:
                response_chat[0]["content"] += metadata["assistant_postfill"].format(**inp_txt)
            if sft_mode == "causal_lm":
                text = apply_chat_template_fn(prompt_chat + response_chat, tokenize=False, add_generation_prompt=False)
                output_texts["text"].append(text)
            elif sft_mode == "completion":
                prompt = apply_chat_template_fn(prompt_chat, tokenize=False, add_generation_prompt=False)
                response = apply_chat_template_fn(response_chat, tokenize=False, add_generation_prompt=False)
                output_texts["prompt"].append(prompt)
                output_texts["response"].append(response)
        return output_texts
        
    return f if not is_intx_model else f_intx
