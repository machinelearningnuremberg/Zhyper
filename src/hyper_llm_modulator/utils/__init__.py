import json
import logging
import os
from math import sqrt

import numpy as np
import torch
import yaml

from .metric_fns import METRIC_FNS
from .model_loading import (
    get_model,
    get_model_and_tokenizer,
    get_tokenizer,
    # load_steering_vec,
    # load_steering_vec_to_model,
    get_peft_config,
)
from .task_metadata import get_metadata_for_task, get_metadata
from .preprocessing import (
    get_preprocessing_fn,
    add_full_stop,
    get_prompt_formatting_fn,
    preprocess_result,
    apply_sfr_template,
)
from .lora_formatting import (
    convert_qkv_gate_up_lora_to_splits_vllm,
    lora_state_dict_to_tensor_dict,
    get_peft_module_names,
    save_lora_from_peft_model,
    get_target_lora_dirs,
    lora_tensor_dict_to_state_dict,
    get_mean_lora,
    get_std_lora,
)
from .pooling import get_pooling_fn
from .utils import embed_texts, embed_tokens, get_inp_tokenize_fn, log_scalar


def repeat_iterator(iterable):
    # infinitely repeat the iterator
    while True:
        yield from iterable


def get_layers(model):
    # Unwrap DDP-wrapped models
    if hasattr(model, "module"):
        return get_layers(model.module)
    if hasattr(model, "model"):
        return get_layers(model.model)
    return model.layers

def get_layers_from_args(args, model):
    len_layers = len(get_layers(model))
    all_layers = range(len_layers)
    if not hasattr(args, 'layers'):
        return all_layers
    if args.layers == None or args.layers == "all":
        return all_layers
    if "last_" in args.layers:
        val = int(args.layers.split("last_")[-1])
        return range(len_layers - val, len_layers)
    if "first_" in args.layers:
        val = int(args.layers.split("first_")[-1])
        return range(0, val)
    if args.layers == "every_4th":
        return range(0, len_layers, 4)


def get_num_params(model):
    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()

    return total_params, trainable_params


def compute_scaling_factor(peft_config):
    if (peft_config is None) or (not hasattr(peft_config, "lora_alpha")):
        return 1.0
    scaling = peft_config.lora_alpha / peft_config.r

    if getattr(peft_config, "use_rslora", False):
        scaling *= sqrt(peft_config.r)

    return scaling


def create_logger(log_dir, debug=False):
    """Create a global logger that logs INFO level messages to stdout and DEBUG ones to debug.log"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("")
    log_formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream_level = logging.DEBUG if debug else logging.INFO
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(stream_level)
    logger.addHandler(stream_handler)

    log_path = f"{log_dir}/debug.log"
    debug_handler = logging.FileHandler(log_path, delay=True)
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(log_formatter)
    logger.addHandler(debug_handler)
    logger.setLevel(logging.DEBUG)
    logger.info(f"Logging to: {log_path}")
    return logger


def save_yaml(data, path):
    with open(path, "w") as file:
        yaml.dump(data, file)


def save_json(data, path):
    with open(path, "w") as file:
        json.dump(data, file, indent=4)


def generate_simplex_points(n_points, dimension):
    # Generate points from a Dirichlet distribution
    points = np.random.dirichlet(np.ones(dimension), size=n_points)

    return torch.tensor(points, dtype=torch.float)


def get_end_points(dimension):
    return torch.eye(dimension, dtype=torch.float)

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
    "australia": "australian",
    "canada": "canadian",
    "thailand": "thai",
    "lebanon": "labenese",
    "saudiarabia": "saudi",
    "pakistan": "pakistani",
    "ukraine": "ukranian",
    "nigeria": "nigerian",
    "brasil": "brazilian",
    "hongkong": "hongkonger",
    "peru": "peruvian",
    "indonesia": "indonesian",
    "spain": "spanish"
}

def _parse_subreddit_name(subreddit_name):
    # gets the subredit postfix 
    # AskAnAfrica = african
    # AskAChinese = chinese
    # PERU = peru
    subreddit_name = subreddit_name.lower()
    if "ask" in subreddit_name:
        if "askan" in subreddit_name:
            subreddit_postfix = subreddit_name.split("askan")[-1]
        elif "aska" in subreddit_name and "asia" not in subreddit_name and "argentina" not in subreddit_name:
            subreddit_postfix =  subreddit_name.split("aska")[-1]
        # elif subreddit_name[-1] == "s":
        #     subreddit_postfix =  subreddit_name.split("ask")[-1][: -1]
        else:
            subreddit_postfix =  subreddit_name.split("ask")[-1]
    else:
        subreddit_postfix = subreddit_name
    return subreddit_postfix

def subreddit_to_nationality(subreddit_name):
    subreddit_name = subreddit_name.lower()
    subreddit_postfix = _parse_subreddit_name(subreddit_name)
    if subreddit_postfix not in ["asia", "argentina"] and ("aska" in subreddit_name):
        # case where subreddit name starts with AskA or AskAn
        nationality = subreddit_postfix
    else:
        nationality = country2nationality[subreddit_postfix]
    return nationality.title()

# reverse
def subreddit_to_country(subreddit_name, map_to_metada_names=False):
    nationality2country = {v:k for k, v in country2nationality.items()}
    subreddit_name = subreddit_name.lower()
    nat = subreddit_to_nationality(subreddit_name)
    country = nationality2country[nat.lower()].title()
    # cases when the country name on subreddit is different:
    if country == "Brasil":
        country = "Brazil"
    elif country == "Saudiarabia":
        country = "Saudi Arabia"
    elif country == "Ph":
        country = "Philippines"
    elif country == "Hongkong":
        country = "Hong Kong"
    if map_to_metada_names:
        if country == "Hong Kong":
            country = "HongKong"
        if country == "Philippines":
            country = "PH"
        if country == "Saudi Arabia":
            country = "SaudiArabia"
        if country == "Uk":
            country = "UK"
        elif country == "America":
            country = "US"
        elif country == "Middleeast":
            country = "MiddleEast"
        else: 
            country = country.title()
    return country
