from copy import deepcopy
import gc
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
    get_num_params,
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

MODEL_INPUT_KEYS = ["input_ids", "attention_mask"]


def log_num_train_params(model):
    logger.debug("Trainable model parameters:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            logger.debug(f"{name}, dtype:{p.dtype}")

    num_total_params, num_trainable_params = get_num_params(model)
    logger.info(
        f"trainable params: {num_trainable_params:,d} "
        f"|| all params: {num_total_params:,d} "
        f"|| trainable%: {100 * num_trainable_params / num_total_params:.4f}"
    )


def main(args):
    args.train_ds_names = args.train_ds_names[: args.n_train_ds]
    args.use_hypernet = use_hypernet = "hyper" in args.exp_setup
    # get task metadata and save to the corresponding run folder
    train_metadata = get_metadata(args.train_ds_names, args.use_per_task_emb)
    val_metadata = get_metadata(args.eval_ds_info, args.use_per_task_emb)
    save_dir = args.save_dir
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
    save_yaml(vars(args), f"{save_dir}/args.yaml")
    set_seed(args.seed)
    # load peft config
    peft_config = None
    peft_type = args.exp_setup.split("_")[-1]
    if peft_type == "lora":
        # used for both normal training and hypernet training
        # for hypernet, the init weights will be copied as the output bias
        peft_config = get_peft_config(
            args.model_dir,
            peft_type,
            target_modules=args.target_modules,
        )
        peft_config.save_pretrained(save_dir)
        logger.debug(f"peft_config:\n{peft_config}")
    else:
        logger.warning(
            "=" * 60 + f"\npeft_type: {peft_type}. Doing normal full-finetuning without any PEFT.\n" + "=" * 60
        )

    # setup accelerator
    plugin = GradientAccumulationPlugin(num_steps=args.grad_accum_steps, sync_with_dataloader=False)
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_plugin=plugin,
        split_batches=True,  # True means do not multiply batch size by the number of gpus used
        log_with="wandb",
    )
    accelerator.seed = args.seed # TODO: does it even work?


    def clear_mem():
        nonlocal accelerator
        torch.cuda.empty_cache()
        accelerator.free_memory()
        gc.collect()

    wandb_dir = f"{os.environ['HOME']}/.wandb/logs/{os.environ['WANDB_PROJECT']}/"
    os.makedirs(wandb_dir, exist_ok=True)
    accelerator.init_trackers(
        os.getenv("WANDB_PROJECT"),
        config=vars(args),
        init_kwargs=dict(wandb={"group": args.run_name, "name": args.run_name, "dir": wandb_dir, "notes": args.notes}),
    )
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
    )
    # train to output delta_w for all layers
    layer_indices = torch.tensor(range(len(get_layers(model))), dtype=torch.long, device=device)
    # NOTE: the module_names is needed for saving generated lora or vera to lora format
    # this is required for using vllm during evaluation
    is_intx_model = tokenizer.chat_template is not None
    assert is_intx_model, "Only chat models are supported"
    logger.debug(f"Model config: {model.config}")
    logger.debug(f"Model: {model}")
    logger.debug(f"is_intx_model: {is_intx_model}")
    # logger.debug(f"Tokenizer: {tokenizer}")
    logger.debug(f"layer_indices: {layer_indices}")

    ##############################################################################
    # emb model and hypermod
    ##############################################################################
    emb_model = None
    emb_tokenizer = None
    task_desc_format_fn = None
    use_explicit_emb_model = False
    pooling_fn = None

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
            model.set_adapter("default")
    elif "lora" in args.exp_setup:
        # for training vanilla LoRA
        model.set_adapter("default")
    else:
        model.train()

    log_num_train_params(model)

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
    # test TODO: remove 
    # if accelerator.is_main_process():
    #     eval_hypermod_checkpoint("/hnvme/workspace/b250be18-hf_helma_1/HyperAlign/train_outputs/sft/z_hyper_lora/20250822-163904_mc2tgWfF/hypermod.pt", 
    #     "/hnvme/workspace/b250be18-hf_helma_1/HyperAlign/train_outputs/sft/z_hyper_lora/20250822-163904_mc2tgWfF/hypermod.pt", 
    #     accelerator.device, 
    #     16, full_eval=True)

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

    uuid = "".join([random.choice(string.ascii_letters + string.digits) for _ in range(8)])
    args.run_name = time.strftime("%Y%m%d-%H%M%S") + f"_{uuid}"
    args.save_dir = f"/hnvme/workspace/b250be18-hf_helma_1/HyperAlign/train_outputs/sft/{args.exp_setup}/{args.run_name}"
    global logger
    # TODO: maybe move somewhere else so that logger doesnt save x GPUs amount of times debug.log.
    logger = create_logger(args.save_dir, debug=args.debug)
    logger.debug(f"CMD: {' '.join(os.sys.argv)}")
    logger.debug(f"args: {args}")
    logger.debug(f"Is CUDA available: {torch.cuda.is_available()}")
    logger.debug(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    main(args)
    subprocess.run("wandb sync --no-include-online --clean", shell=True)
    subprocess.run("wandb artifact cache cleanup 10GB", shell=True)

# from sacred import Experiment
# from types import SimpleNamespace
# from seml.commands import start
# from seml.settings import SETTINGS

# def get_experiment_environment_clean_env(experiment):
#     if exp_env := experiment['seml'].get('env'):
#         env = dict(exp_env)
#         # env = {"SLURM_EXPORT_ENV": "ALL"}
#         print("here")
#         print(env)
#     env = {**env, **SETTINGS.EXPERIMENT.ENVIRONMENT}
#     return env

# start.get_experiment_environment = get_experiment_environment_clean_env


# ex = Experiment()


# @ex.config
# def config():
#     exp_setup = "hyper_lora"
#     model_dir = "google/gemma-2-2b-it"
#     emb_model = "Alibaba-NLP/gte-large-en-v1.5"
#     warmup_frac = 0.2
#     lr = 2.5e-5
#     n_tasks_per_batch = 8
#     n_points_per_task = 1
#     grad_accum_steps = 1
#     epochs = 20000
#     n_descs_per_ds = 128
#     n_train_ds = 479
#     encoder_type = "linear"
#     l2_reg_generated_w = 1e-3
#     label_smoothing = 0.1
#     neftune_noise_alpha = 5
#     weight_decay = 1e-2
#     hypernet_latent_size = 512
#     head_in_size = 2048
#     val_batch_size = 32
#     seed = 42
#     debug = False
#     notes = "test"
#     run_name = "test"
#     save_dir = "train_outputs/sft/test"
#     train_ds_names = ["train"]  # TODO: add train ds names
#     eval_ds_info = {"train": {"name": "train", "n_descs": 128}}  # TODO: add eval ds info
#     use_per_task_emb = True
#     use_one_hot_task_emb = False
#     use_inp_as_desc = False
#     use_per_sample_desc = False
#     use_default_desc = False
#     use_hypernet = True
#     use_explicit_emb_model = False
#     use_explicit_hypermod = False
#     use_hypermod_as_emb = False
#     use_hypermod_as_desc = False
#     use_hypermod_as_inp = False
#     use_hypermod_as_out = False
#     use_hypermod_as_none = False
#     use_hypermod_as_all = False

# @ex.automain
# def main_seml(_config):
#     args = SimpleNamespace(**_config)
#     os.environ["TOKENIZERS_PARALLELISM"] = "true"
#     os.environ["WANDB_MODE"] = "disabled"
#     os.environ["WANDB_PROJECT"] = "hypermod_sft"
#     os.environ["WANDB_WATCH"] = "all"
#     os.environ["WANDB_CONSOLE"] = "off"
#     os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#     disable_caching()

#     assert (
#         args.use_per_task_emb + args.use_inp_as_desc + args.use_per_sample_desc + args.use_default_desc
#     ) <= 1, "only one or none of use_per_task_emb, use_inp_as_desc, use_per_sample_desc can be used"

#     assert (
#         args.use_per_task_emb or not args.use_one_hot_task_emb
#     ), "one_hot_task_emb can only be used with use_per_task_emb"

#     uuid = "".join([random.choice(string.ascii_letters + string.digits) for _ in range(8)])
#     args.run_name = time.strftime("%Y%m%d-%H%M%S") + f"_{uuid}"
#     args.save_dir = f"train_outputs/sft/{args.exp_setup}/{args.run_name}"
#     global logger
#     logger = create_logger(args.save_dir, debug=args.debug)
#     logger.debug(f"CMD: {' '.join(os.sys.argv)}")
#     logger.debug(f"args: {args}")
#     logger.debug(f"Is CUDA available: {torch.cuda.is_available()}")
#     logger.debug(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

#     main(args)
#     subprocess.run("wandb sync --no-include-online --clean", shell=True)
#     subprocess.run("wandb artifact cache cleanup 10GB", shell=True)
