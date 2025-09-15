import argparse
import os
import sys
import time
import random
import string

import torch
import yaml
from peft import get_peft_config, load_peft_weights, PeftConfig

from hyper_llm_modulator.utils import (
    get_layers,
    embed_texts,
)
from hyper_llm_modulator.hyper_modulator import (
    HyperModulator,
    load_hypermod_checkpoint,
    save_lora,
)
from hyper_llm_modulator.utils.model_loading import get_emb_model_and_fns


def add_full_stop(s):
    s = s.strip()
    # check if s ends with . or .*
    if s[-1].isalpha():
        s += "."
    return s


def load_hypermod(hypermod_dir, device):
    checkpoint_path = f"{hypermod_dir}/hypermod.pt"
    (
        args,
        hypermod,
        model,
        tokenizer,
        emb_model,
        emb_tokenizer,
        task_desc_format_fn,
        pooling_fn,
    ) = load_hypermod_checkpoint(checkpoint_path, device)
    return (
        args,
        hypermod,
        model,
        tokenizer,
        emb_model,
        emb_tokenizer,
        task_desc_format_fn,
        pooling_fn,
    )

country2nationality = {
    "egypt": "egyptian",
    "europe": "european",
    "asia": "asian",
    "italy": "italian",
    "india": "indian",
    "latinamerica": "latin american",
    "mexico": "mexican",
    "middleeast": "middle eastern",
    "southafrica": "south african",
    "turkey": "turkish",
    "uk": "british",
    "ph": "filipino",
    "argentina": "argentinian",
    "germany": "german",
    "china": "chinese",
    "japan": "japanese",
    "africa": "african",
    "america": "american",
    "russia": "russian",
    "balkans": "balkan",
    "france": "french",
}

def prcoess_subreddit_names_reverse(subreddit_name):
    nat2country = {v:k for k, v in country2nationality.items()}
    subreddit_name = subreddit_name.lower()
    nat = ""
    if "askan" in subreddit_name:
        nat = subreddit_name.split("askan")[-1]
    elif "aska" in subreddit_name and "asia" not in subreddit_name and "argentina" not in subreddit_name:
        nat =  subreddit_name.split("aska")[-1]
    elif subreddit_name[-1] == "s":
        nat =  subreddit_name.split("ask")[-1][: -1]
    else:
        nat =  country2nationality[subreddit_name.split("ask")[-1]]
    return nat2country[nat].title()

if __name__ == "__main__":
    DATA_DIR="/home/hpc/b250be/b250be18/HyperAlignz/cul_data/descriptions_commands"
    files = os.listdir(DATA_DIR)
    for file in files:
        file_path = os.path.join(DATA_DIR, file)
        subreddit = file.replace("_descriptions_commands.yaml", "")
        country = prcoess_subreddit_names_reverse(subreddit)
        with open(file_path, "r") as file:
            rand_cond = yaml.safe_load(file)[0]
        default_hypermod_dir = "/hnvme/workspace/b250be18-hf_helma_1/HyperAlign/train_outputs/sft/z_hyper_lora/20250911-144303_KOCReKeU"
        # default_task_desc   = "You are a German person. A German is someone who values order and punctuality, often showing a strong sense of responsibility in daily life."
        default_task_desc = rand_cond
        # default_exp_setup   = "hyper_lora"
        # TODO add index here for one hot
        ds_one_hot = None

        # Use args if available, otherwise fallback
        hypermod_dir = sys.argv[1] if len(sys.argv) > 1 else default_hypermod_dir
        task_desc    = sys.argv[2].strip("\"' ") if len(sys.argv) > 2 else default_task_desc

        print(f"\nGenerating LoRA for description:\n\n{task_desc}")

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # load metadata
        args = argparse.Namespace(**yaml.safe_load(open(f"{hypermod_dir}/args.yaml", "r")))
        peft_config = get_peft_config(
            PeftConfig.from_json_file(f"{hypermod_dir}/adapter_config.json")
        )
        # curtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        # uuid = "".join(
        #     [random.choice(string.ascii_letters + string.digits) for _ in range(8)]
        # )
        # save_name = f"{curtime}_{uuid}"
        save_name = f"{country}"
        (
            args,
            hypermod,
            model,
            tokenizer,
            emb_model,
            emb_tokenizer,
            task_desc_format_fn,
            pooling_fn,
        ) = load_hypermod(hypermod_dir, device)
        layer_indices = range(len(get_layers(model)))
        layer_indices = torch.tensor(layer_indices, dtype=torch.long, device=device)
        emb_size = emb_model.config.hidden_size

        # generate loras
        if False:
            # TODO
            eye = torch.eye(len(args.train_ds_names)).to(device)
            train_idx = args.train_ds_names.index(ds_one_hot)
            task_emb = eye[train_idx].unsqueeze(0)
            # task_emb = task_emb.unsqueeze(0)
        else:
            task_emb = embed_texts(
                [task_desc], emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn, device
            )
        encoder_out = hypermod.task_encoder(task_emb)
        encoded_task_emb = encoder_out["encoded_task_emb"].detach()
        lora_sd = hypermod.gen_lora(layer_indices, encoded_task_emb, model) # TODO: add model
        lora_dir = f"{hypermod_dir}/extras/user_generated/{save_name}/"
        save_lora(lora_sd, peft_config, lora_dir)
        with open(f"{lora_dir}/task_desc.txt", "w") as f:
            f.write(task_desc)
        print(f"Saved lora to {lora_dir}")
