import logging
from math import sqrt
import os

import torch
from peft import PeftModel
from peft import get_peft_config as _get_peft_config
from peft.utils import PeftType
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

from hyper_llm_modulator.utils.pooling import get_pooling_fn
from hyper_llm_modulator.utils.preprocessing import add_full_stop, apply_sfr_template

logger = logging.getLogger()


def get_model_and_tokenizer(
    model_path,
    train,
    requires_grad,
    use_flash_attn=True,
    peft_config=None,
    model_kwargs=None,
    tokenizer_kwargs=None,
    device="cuda:0",
    dtype=torch.bfloat16,
    peft_adapter_path=None
):
    model = get_model(
        model_path,
        train,
        requires_grad,
        use_flash_attn,
        peft_config,
        model_kwargs,
        device,
        dtype,
        peft_adapter_path
    )
    tokenizer = get_tokenizer(model_path, tokenizer_kwargs, peft_config, train)
    return model, tokenizer


def get_tokenizer(model_path, tokenizer_kwargs=None, peft_config=None, train=False):
    # NOTE: lora models don't have tokenizer config in the folder

    padding_side = "left" if not train else "right"
    if peft_config:
        model_path = peft_config.base_model_name_or_path

    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding_side=padding_side, **tokenizer_kwargs
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    template_path = f"chat_templates/{model_path}/chat_template.jinja"
    assert os.path.exists(template_path), (
        f"Chat template not found for {model_path}.\n"
        "We assume a specfic form of chat template for consistency between models. "
        "Please use the templates provided."
    )
    print(f"Loading chat template from {template_path}")
    chat_template = open(template_path).read()
    chat_template = chat_template.replace("    ", "").replace("\n", "")
    tokenizer.chat_template = chat_template

    tokenizer.add_eos_token = False
    tokenizer.truncation_side = "left"
    return tokenizer


def get_model(
    model_path,
    train,
    requires_grad,
    use_flash_attn=True,
    peft_config=None,
    model_kwargs=None,
    device="cuda:0",
    dtype=torch.bfloat16,
    peft_adapter_path=None
):
    model_init_kwargs = dict(
        pretrained_model_name_or_path=model_path,
        device_map=device,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if model_kwargs is not None:
        model_init_kwargs.update(model_kwargs)
    if use_flash_attn:
        model_init_kwargs["attn_implementation"] = "flash_attention_2"
    if train:
        # for training disable cache
        model_init_kwargs["use_cache"] = False
    logger.debug(f"Model init kwargs: {model_init_kwargs}")
    model = AutoModelForCausalLM.from_pretrained(**model_init_kwargs)
    if peft_adapter_path is not None:
        model = PeftModel.from_pretrained(model, peft_adapter_path)
    elif peft_config is not None:
        model = PeftModel(model, peft_config)
    model.train(train)
    for param in model.parameters():
        param.requires_grad = requires_grad
    return model


def get_peft_config(model_dir, peft_type, **kwargs):
    peft_type = peft_type.upper()
    assert peft_type in [PeftType.LORA, PeftType.VERA]

    peft_conf_kwargs = dict(
        r=8 if peft_type == PeftType.LORA else 64,
        peft_type=peft_type,
        base_model_name_or_path=model_dir,
        task_type="CAUSAL_LM",
    )

    peft_conf_kwargs[f"{peft_type.lower()}_dropout"] = 0.05

    if peft_type == PeftType.LORA:
        peft_conf_kwargs["use_rslora"] = True
        peft_conf_kwargs["lora_alpha"] = peft_conf_kwargs["r"] * 2

    peft_conf_kwargs.update(kwargs)
    peft_config = _get_peft_config(peft_conf_kwargs)
    return peft_config


def get_emb_model_and_fns(emb_model_name, device):
    emb_model = AutoModel.from_pretrained(
        emb_model_name,
        device_map=device,
        torch_dtype=torch.float32 if "gte" in emb_model_name else torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    emb_tokenizer = AutoTokenizer.from_pretrained(emb_model_name)
    if emb_tokenizer.pad_token_id is None:
        emb_tokenizer.pad_token_id = emb_tokenizer.eos_token_id
    task_desc_format_fn = add_full_stop
    if "SFR" in emb_model_name:
        task_desc_format_fn = apply_sfr_template
        pooling_fn = get_pooling_fn("last_token")
    elif "gte" in emb_model_name:
        pooling_fn = get_pooling_fn("cls")
    return emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn
