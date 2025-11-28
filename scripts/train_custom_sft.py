from copy import deepcopy
import gc
import logging
import os
import random
import string
import subprocess
import time

import torch
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from datasets import disable_caching
from transformers import set_seed, get_scheduler

from hyper_llm_modulator.configs import ArgumentParser, TrainingArguments
from hyper_llm_modulator.data import create_dataloaders
from hyper_llm_modulator.hyper_modulator import create_hypermod
from hyper_llm_modulator.sft_trainer import train
from hyper_llm_modulator.utils import (
    get_layers,
    create_logger,
    save_yaml,
    get_model_and_tokenizer,
    get_peft_config,
    get_pooling_fn,
    add_full_stop,
    get_metadata,
)
# from hyper_llm_modulator.utils.eval_hypermod import eval_hypermod_checkpoint
from hyper_llm_modulator.utils.model_loading import get_emb_model_and_fns
import os

from hyper_llm_modulator.utils import get_layers_from_args

MODEL_INPUT_KEYS = ["input_ids", "attention_mask"]


def main(args, accelerator: Accelerator):
    set_seed(args.seed)
    if args.n_train_ds < 479:
        random.shuffle(args.train_ds_names)
    args.train_ds_names = args.train_ds_names[: args.n_train_ds]
    args.use_hypernet = use_hypernet = "hyper" in args.exp_setup
    # get task metadata and save to the corresponding run folder
    train_metadata = get_metadata(args.train_ds_names, args.use_per_task_emb, args.ds_type == "align")
    val_metadata = get_metadata(args.eval_ds_info, args.use_per_task_emb, args.ds_type == "align")
    accelerator.wait_for_everyone()
    
    save_dir = args.save_dir
    if accelerator.is_main_process:
        os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
        save_yaml(vars(args), f"{save_dir}/args.yaml")
    accelerator.wait_for_everyone()


    # load peft config
    peft_config = None
    peft_type = None
    if "lora" in args.exp_setup: 
        peft_type = "lora"
    elif "vera" in args.exp_setup: 
        peft_type = "vera"
    if peft_type in ["lora", "vera"]:
        # used for both normal training and hypernet training
        # for hypernet, the init weights will be copied as the output bias
        peft_config = get_peft_config(
            args.model_dir,
            peft_type,
            target_modules=args.target_modules,
            r=args.r
        )
        # Only main process saves the peft config
        if accelerator.is_main_process:
            peft_config.save_pretrained(save_dir)
            logger.debug(f"peft_config:\n{peft_config}")
    else:
        logger.warning(
            "=" * 60 + f"\npeft_type: {peft_type}. Doing normal full-finetuning without any PEFT.\n" + "=" * 60
        )


    def clear_mem():
        nonlocal accelerator
        torch.cuda.empty_cache()
        accelerator.free_memory()
        gc.collect()
    device = accelerator.device

    ##############################################################################
    # Model setup
    ##############################################################################
    model, tokenizer = get_model_and_tokenizer(
        args.model_dir,
        train=True,
        requires_grad=True if "fullfinetune" in args.exp_setup else False,
        peft_config=peft_config,
        model_kwargs={"output_hidden_states": True, "output_attentions": False},
        device=device,
        exp_setup=args.exp_setup
    )
    # train to output delta_w for all layers
    layer_indices = torch.tensor(get_layers_from_args(args, model), dtype=torch.long, device=device)
    # NOTE: the module_names is needed for saving generated lora or vera to lora format
    # this is required for using vllm during evaluation
    is_intx_model = tokenizer.chat_template is not None
    assert is_intx_model, "Only chat models are supported"
    logger.debug(f"Model config: {model.config}")
    logger.debug(f"Model: {model}")
    logger.debug(f"is_intx_model: {is_intx_model}")
    # logger.debug(f"Tokenizer: {tokenizer}")
    logger.debug(f"layer_indices: {layer_indices}")
    logger.info(f"layer_indices: {layer_indices}")

    ##############################################################################
    # emb model and hypermod
    ##############################################################################
    emb_model = None
    emb_tokenizer = None
    task_desc_format_fn = None
    use_explicit_emb_model = False
    pooling_fn = None

    #### LoRA-XS
    # https://github.com/MohammadrezaBanaei/LoRA-XS/blob/main/utils/initialization_utils.py
    if "lora_xs" in args.exp_setup:
        # z_hyper full = hyper lora with lora_xs
        # z_hyper diag = Zhyper with lora_xs
        from peft.utils import _get_submodules
        from hyper_llm_modulator.utils.utils import replace_by_svd, \
                                                    replace_module_weights, \
                                                    forward_latent_lora_xs, get_delta_weight_lora_xs, init_module_weights
        import types
        key_list = [key for key, _ in model.named_modules()]
        for key in key_list:
            target_module_found = any(key.endswith(target_key) for target_key in peft_config.target_modules)
            if target_module_found:
                _, target, _ = _get_submodules(model, key)
                A_SVD, B_SVD = replace_by_svd(weight=target.weight.T, rank=args.r)
                # )
                replace_module_weights(target.lora_B.default, B_SVD.T, renormalize=False)
                replace_module_weights(target.lora_A.default, A_SVD.T, renormalize=False)
    hypermod = None
    if use_hypernet:
        task_emb_size = None
        if not args.use_one_hot_task_emb:
            emb_model = model
            emb_tokenizer = deepcopy(tokenizer)
            task_desc_format_fn = add_full_stop
            pooling_fn = get_pooling_fn("last_token")

            if args.emb_model:
                use_explicit_emb_model = True
                emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn = get_emb_model_and_fns(
                    args.emb_model, device
                )
                logger.debug(f"emb_model: {emb_model}")
            emb_model.eval()
            task_emb_size = emb_model.config.hidden_size
        hypermod = create_hypermod(args, peft_type, device, model, layer_indices, task_emb_size)
        logger.debug(f"Hypermod: {hypermod}")
        model.add_module("hypermod", hypermod)
        if "z" in args.exp_setup:
            # in case of z setting, train both vanilla and hypernet
            model.set_adapter("default")
    elif ("lora" in args.exp_setup) or ("vera" in  args.exp_setup):
        # for training vanilla LoRA/VeRA
        model.set_adapter("default")
    else:
        model.train()

    ##############################################################################
    # Dataset setup
    ##############################################################################
    data_loaders = create_dataloaders(
        args,
        train_metadata,
        val_metadata,
        use_hypernet,
        device,
        tokenizer,
        is_intx_model,
        emb_model,
        emb_tokenizer,
        task_desc_format_fn,
        pooling_fn,
    )
    train_dataloader = data_loaders["train"]
    val_dataloaders = {k: v for k, v in data_loaders.items() if "train" not in k}

    ##############################################################################
    # Training
    ##############################################################################      

    # freezing all lora weights.
    if ("rand" in args.exp_setup) or ("lora_xs" in args.exp_setup): # putting this in model init does not work with DDP.
        for name, param in model.named_parameters():
            if not "hypermod" in name:
                param.requires_grad = False

    if len(layer_indices) != len(get_layers(model)):
        for name, param in model.named_parameters():
            if "lora" in name and "layers" in name:
                layer_idx = int(name.split("layers.")[-1].split(".")[0])
                if layer_idx not in layer_indices:
                    param.requires_grad = False

    wd = args.weight_decay
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=wd)
    model, optimizer = accelerator.prepare(model, optimizer)
    train_dataloader = accelerator.prepare(train_dataloader)
    for k, v in val_dataloaders.items():
        val_dataloaders[k] = accelerator.prepare(v)
    num_training_steps = args.epochs * len(train_dataloader)
    num_warmup_steps = args.warmup_frac * num_training_steps
    scheduler = get_scheduler(
        "linear",
        optimizer,
        num_warmup_steps=int(num_warmup_steps / args.grad_accum_steps),
        num_training_steps=int(num_training_steps / args.grad_accum_steps),
    )
    scheduler = accelerator.prepare(scheduler)
    inp_dropout = getattr(peft_config, f"{peft_type.lower()}_dropout", 0.0)

    if use_explicit_emb_model:
        del emb_model, emb_tokenizer
        clear_mem()

    train(
        args,
        save_dir,
        inp_dropout,
        accelerator,
        model,
        layer_indices,
        hypermod,
        train_dataloader,
        val_dataloaders,
        optimizer,
        num_training_steps,
        scheduler,
        tokenizer
    )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["WANDB_PROJECT"] = "hypermod_sft"
    os.environ["WANDB_WATCH"] = "all"
    os.environ["WANDB_CONSOLE"] = "off"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # os.environ["WANDB_MODE"] = "disabled"
    disable_caching()

    parser = ArgumentParser((TrainingArguments,))
    args = parser.parse()
    assert (
        args.use_per_task_emb + args.use_inp_as_desc + args.use_per_sample_desc + args.use_default_desc
    ) <= 1, "only one or none of use_per_task_emb, use_inp_as_desc, use_per_sample_desc can be used"

    assert (
        args.use_per_task_emb or not args.use_one_hot_task_emb
    ), "one_hot_task_emb can only be used with use_per_task_emb"

    # setup accelerator
    plugin = GradientAccumulationPlugin(num_steps=args.grad_accum_steps, sync_with_dataloader=False)
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_plugin=plugin,
        split_batches=True,  # True means do not multiply batch size by the number of gpus used
        log_with="wandb",
    )
    accelerator.seed = args.seed

    global logger
    uuid = "".join([random.choice(string.ascii_letters + string.digits) for _ in range(8)])
    if args.run_name is None:
        args.run_name = time.strftime("%Y%m%d-%H%M%S") + f"_{uuid}"
    if args.save_dir is None:
        args.save_dir = f"/train_outputs/sft/{args.exp_setup}/{args.run_name}"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process: 
        logger = create_logger(args.save_dir, debug=args.debug)
    accelerator.wait_for_everyone()
    logger = logging.getLogger("")
    wandb_dir = f"{os.environ['HOME']}/.wandb/logs/{os.environ['WANDB_PROJECT']}/"
        # os.makedirs(wandb_dir, exist_ok=True)
    accelerator.init_trackers(
        os.getenv("WANDB_PROJECT"),
        config=vars(args),
        init_kwargs=dict(wandb={"group": args.run_name, "name": args.run_name, "dir": wandb_dir, "notes": args.notes}),
    )

    logger.debug(f"CMD: {' '.join(os.sys.argv)}")
    logger.info(f"args: {args}")
    logger.debug(f"Is CUDA available: {torch.cuda.is_available()}")
    logger.debug(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    main(args, accelerator)
    subprocess.run("wandb sync --no-include-online --clean", shell=True)
    subprocess.run("wandb artifact cache cleanup 10GB", shell=True)
