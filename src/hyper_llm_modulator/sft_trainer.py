from collections import defaultdict
from contextlib import contextmanager
from glob import glob
import logging
import os
from functools import partial
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from accelerate.utils import broadcast

import torch
import wandb
from peft import PeftModel
from transformers.modeling_utils import unwrap_model

from hyper_llm_modulator.hooks import add_lora_hooks, remove_hook_handles_
from hyper_llm_modulator.hyper_modulator import get_init_peft_weights, get_peft_weights, save_hypermod_checkpoint
from hyper_llm_modulator.utils import save_lora_from_peft_model, log_scalar, get_num_params
import subprocess

# from hyper_llm_modulator.utils.eval_hypermod import eval_lora

logger = logging.getLogger("")

MODEL_INPUT_KEYS = ["input_ids", "attention_mask"]


# taken from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# taken from https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/3
@contextmanager
def evaluating(*models):
    """Temporarily switch to evaluation mode."""
    is_training = [model.training if model is not None else False for model in models]
    try:
        for model in models:
            if model is not None:
                model.eval()
        yield models
    finally:
        for model, training in zip(models, is_training):
            if model is not None:
                model.train(training)


def neftune_post_forward_hook(module, input, output):
    """
    Implements the NEFTune forward pass for the model using forward hooks. Note this works only for
    torch.nn.Embedding layers. This method is slightly adapted from the original source code
    that can be found here: https://github.com/neelsjain/NEFTune

    Simply add it to your model as follows:
    ```python
    model = ...
    model.embed_tokens.neftune_noise_alpha = 0.1
    model.embed_tokens.register_forward_hook(neftune_post_forward_hook)
    ```

    Args:
        module (`torch.nn.Module`):
            The embedding module where the hook is attached. Note that you need to set
            `module.neftune_noise_alpha` to the desired noise alpha value.
        input (`torch.Tensor`):
            The input tensor to the model.
        output (`torch.Tensor`):
            The output tensor of the model (i.e. the embeddings).
    """
    if module.training:
        dims = torch.tensor(output.size(1) * output.size(2))
        mag_norm = module.neftune_noise_alpha / torch.sqrt(dims)
        output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
    return output


def trl_activate_neftune(model, neftune_noise_alpha):
    r"""
    Activates the neftune as presented in this code: https://github.com/neelsjain/NEFTune and paper: https://arxiv.org/abs/2310.05914
    Since in transformers Trainer we do have an `_activate_neftune` method, we need to rename this method to avoid conflicts.
    """
    unwrapped_model = unwrap_model(model)
    if isinstance(unwrapped_model, PeftModel):
        embeddings = unwrapped_model.base_model.model.get_input_embeddings()
    else:
        embeddings = unwrapped_model.get_input_embeddings()

    embeddings.neftune_noise_alpha = neftune_noise_alpha
    hook_handle = embeddings.register_forward_hook(neftune_post_forward_hook)
    return hook_handle


def get_loss_batch(
    batch,
    model,
    target_modules,
    inp_dropout,
    layer_indices,
    use_hypernet,
    hypermod,
    equally_weight_sample,
    l2_reg_generated_w=0,
    label_smoothing=0,
    return_per_token_acc=False,
    return_entropy=False,
):
    out = dict()
    out["generated_w_l2_loss"] = torch.zeros(1, device=model.device)
    bs = batch["input_ids"].shape[0]
    hook_handles = []
    if use_hypernet:
        # TODO: allow online embed of hypernetwork's input
        # to support hyperdecoders style training
        # (using the input prompt as the task description)
        encoder_out = hypermod.task_encoder(batch["task_embs"])
        encoded_task_emb = encoder_out["encoded_task_emb"]
        # generated lora weights only once for all samples
        # then hook the generated loras to the model
        factorized_delta_w, hook_handles = generate_and_hook_delta_w(
            target_modules=target_modules,
            inp_dropout=inp_dropout,
            model=model,
            layer_indices=layer_indices,
            hypermod=hypermod,
            encoded_task_emb=encoded_task_emb,
            bs=bs,
            training=model.training,
        )
        if l2_reg_generated_w:
            # Original A/B regularization
            for A, B in factorized_delta_w.values():
                out["generated_w_l2_loss"] += ((A**2).mean() + (B**2).mean()) * l2_reg_generated_w
    outputs = model(**{k: batch[k] for k in MODEL_INPUT_KEYS})
    out["sft_loss"] = compute_loss(
        batch["labels"],
        outputs.logits,
        equally_weight_sample=equally_weight_sample,
        label_smoothing=label_smoothing,
    )
    if return_per_token_acc or return_entropy:
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        indices = torch.where(shift_labels != -100)
    if return_per_token_acc:
        # only compute acc when batch["labels"] != -100
        out["per_token_acc"] = (shift_logits.argmax(-1) == shift_labels)[indices].float().mean()
    if return_entropy:
        logits = shift_logits[indices]
        prob = torch.nn.functional.softmax(logits, dim=-1)
        out["entropy"] = -torch.sum(prob * torch.log(prob), dim=-1).mean()
    remove_hook_handles_(hook_handles)
    return out


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

def _get_percentage_value(s: str):
    if not isinstance(s, str):
        return None
    
    pattern = re.compile(r"^\s*(\d+(\.\d+)?)%\s*$")
    match = pattern.match(s)
    if match:
        return float(match.group(1)) / 100
    return None


def train(
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
):

    torch.autograd.set_detect_anomaly(True)
    model.train()
    if args.use_hypernet:
        hypermod.train()
        if accelerator.is_main_process:
            wandb.watch(hypermod, log="all", log_freq=1000)
    log_num_train_params(model)

    _log_train_vals = partial(
        log_train_vals,
        len_train_dataloader=len(train_dataloader),
        scheduler=scheduler,
        accelerator=accelerator
    )

    _get_loss_batch = partial(
        get_loss_batch,
        model=model,
        target_modules=args.target_modules,
        inp_dropout=inp_dropout,
        layer_indices=layer_indices,
        use_hypernet=args.use_hypernet,
        hypermod=hypermod,
        equally_weight_sample=args.equally_weight_sample,
    )
    _get_loss_batch_train = partial(
        _get_loss_batch,
        label_smoothing=args.label_smoothing,
        l2_reg_generated_w=args.l2_reg_generated_w,
    )

    neftune_hook_handle = trl_activate_neftune(model, args.neftune_noise_alpha)

    ##########################################
    # Training
    ##########################################
    # validate before training
    if args.also_val_on_train:
        val_info = validate(model, hypermod, {"train": train_dataloader}, _get_loss_batch, curstep=0)
    val_info = validate(model, hypermod, val_dataloaders, _get_loss_batch, curstep=0)
    if args.use_hypernet:
        cp_path = save_hypermod_checkpoint(save_dir, hypermod, curstep=0, accelerator=accelerator)
        if "z" in args.exp_setup:
            lora_dir = save_lora_checkpoint(save_dir, accelerator.unwrap_model(model), args.model_dir, curstep=0, accelerator=accelerator)
    elif ("mt_lora" in args.exp_setup) or ("mt_vera" in args.exp_setup):
        lora_dir = save_lora_checkpoint(save_dir, accelerator.unwrap_model(model), args.model_dir, curstep=0, accelerator=accelerator)
    elif "val/seen" in val_info:
        # normal LoRA training
        stopper = EarlyStopper(patience=3, min_delta=0)
        stopper.early_stop(val_info["val/seen"]["sft_loss"])

    # if accelerator.is_main_process:
    #     if os.environ.get("SLURM_JOB_ID", None) is not None:
    #         # here we run the watcher on demand when on slurm to save resources.
    #         sbatch_res = subprocess.run(["sbatch", "scripts/watcher.sh", "false", "true"], capture_output=True, text=True)
    #         logging.info(sbatch_res)


    curstep = 1
    grad_norm = 0
    avg_losses = defaultdict(list)
    early_stop = False

    pct_gen_freq = _get_percentage_value(args.gen_freq)
    if pct_gen_freq is not None:
        gen_freq = max(1, int(num_training_steps * pct_gen_freq))
    else:
        gen_freq = int(args.gen_freq)
    pct_logging_freq = _get_percentage_value(args.logging_freq)
    if pct_logging_freq is not None:
        logging_freq = max(1, int(num_training_steps * pct_logging_freq))
    else:
        logging_freq = int(args.logging_freq)

    pct_val_freq = _get_percentage_value(args.val_freq)
    if pct_val_freq is not None:
        val_freq = max(1, int(num_training_steps * pct_val_freq))
    else:
        val_freq = int(args.val_freq)
    
    for _ in (pbar := tqdm(range(args.epochs), total=num_training_steps)):
        for batch in train_dataloader:
            with accelerator.accumulate(model), accelerator.autocast():
                # print(tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False))
                batch_loss = _get_loss_batch_train(batch)
                loss = batch_loss["sft_loss"] + batch_loss["generated_w_l2_loss"]
                avg_losses["train/sft_loss"].append(batch_loss["sft_loss"].item())
                avg_losses["train/generated_w_l2_loss"].append(batch_loss["generated_w_l2_loss"].item())
                avg_losses["train/total_loss"].append(loss.item())

                optimizer.zero_grad()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()

            pbar.update(1)
            pbar.set_description(f"loss: {loss.item():.4f}")
            ##########################################
            # Logging and Validation
            ##########################################
            if (curstep % gen_freq == 0) or (curstep == num_training_steps):
                bs = batch["input_ids"].shape[0]
                num_show = 5 if bs >= 5 else bs
                sample_indices = torch.randperm(bs)[:num_show].tolist()
                for i in sample_indices:
                    ids = batch["input_ids"][i]
                    labels = batch["labels"][i]
                    # prompt tokens are where labels == -100
                    prompt_ids = ids[labels == -100]
                    prompt_ids = prompt_ids[prompt_ids != tokenizer.pad_token_id]
                    tokenizer.padding_side = "left"
                    # hacky way to get the template from the ids.
                    if "Mistral" in tokenizer.name_or_path:
                        prompt_ids = torch.cat([prompt_ids, tokenizer.encode(" ", add_special_tokens=False, return_tensors="pt").squeeze(0).to(prompt_ids.device)])
                    else:
                        messages = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "Tell me"},
                        ]

                        ids_with_prompt = tokenizer.apply_chat_template(
                            messages, add_generation_prompt=True, tokenize=True
                        )
                        ids_without_prompt = tokenizer.apply_chat_template(
                            messages, add_generation_prompt=False, tokenize=True
                        )
                        gen_marker_ids = ids_with_prompt[len(ids_without_prompt):]
                        prompt_ids = torch.cat([
                            prompt_ids,
                            torch.tensor(gen_marker_ids, dtype=prompt_ids.dtype, device=prompt_ids.device)
                        ])
                    with torch.no_grad(), evaluating(model):
                        unwrapped = accelerator.unwrap_model(model)
                        original_use_cache = getattr(unwrapped.config, "use_cache", None)
                        if original_use_cache is not True:
                            try:
                                unwrapped.config.use_cache = True
                            except Exception:
                                pass
                        input_ids = prompt_ids.unsqueeze(0)
                        attention_mask = torch.ones_like(input_ids)
                        outputs = unwrapped.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=256,
                            do_sample=True,
                            # temperature=1.0,
                            # top_p=1.0,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id
                        )
                        if original_use_cache is not None and original_use_cache is not True:
                            try:
                                unwrapped.config.use_cache = original_use_cache
                            except Exception:
                                pass
                    logging.info(tokenizer.decode(outputs[0], skip_special_tokens=True))
                tokenizer.padding_side = "right"
            if (curstep % logging_freq == 0) or (curstep == num_training_steps):
                _log_train_vals(grad_norm, avg_losses, curstep)
                # reset avg_losses
                avg_losses = defaultdict(list)
            if (curstep % val_freq == 0) or (curstep == num_training_steps):
                if args.also_val_on_train:
                    val_info = validate(model, hypermod, {"train": train_dataloader}, _get_loss_batch, curstep)
                val_info = validate(model, hypermod, val_dataloaders, _get_loss_batch, curstep)
                if args.use_hypernet:
                    cp_path = save_hypermod_checkpoint(save_dir, hypermod, curstep, accelerator)
                    if "z" in args.exp_setup:
                        lora_dir = save_lora_checkpoint(save_dir, accelerator.unwrap_model(model), args.model_dir, curstep, accelerator)
                elif ("mt_lora" in args.exp_setup) or ("mt_vera" in args.exp_setup):
                    lora_dir = save_lora_checkpoint(save_dir, accelerator.unwrap_model(model), args.model_dir, curstep, accelerator)
                elif "val/seen" in val_info:
                    # Compute early stop on main process, but ALL ranks must call broadcast
                    should_stop_main = False
                    if accelerator.is_main_process:
                        if stopper.early_stop(val_info["val/seen"]["sft_loss"]):
                            logger.info("Early stopping")
                            should_stop_main = True
                    stop_signal = torch.tensor(int(should_stop_main), device=accelerator.device)
                    stop_signal = broadcast(stop_signal, from_process=0)
                    early_stop = bool(stop_signal.item())

                # read early stop signal from the watcher
                has_earlystop_file_main = False
                if accelerator.is_main_process:
                    has_earlystop_file_main = os.path.isfile(f"{save_dir}/earlystop_info.yaml")
                stop_signal = torch.tensor(int(has_earlystop_file_main), device=accelerator.device)
                stop_signal = broadcast(stop_signal, from_process=0)
                if bool(stop_signal.item()):
                    early_stop = True

            curstep += 1
            if early_stop:
                break
        if early_stop:
            break
    # accelerator.wait_for_everyone()
    if args.use_hypernet:
        last_cp_path_hypermod = save_hypermod_checkpoint(save_dir, hypermod, curstep, accelerator)
        if accelerator.is_main_process:
            best_cp_path_hypermod = f"{save_dir}/hypermod.pt"
            if not os.path.isfile(best_cp_path_hypermod):
                shutil.copy(last_cp_path_hypermod, f"{save_dir}/hypermod.pt")
        if "z" in args.exp_setup:
            lora_dir = save_lora_checkpoint(save_dir, accelerator.unwrap_model(model), args.model_dir, curstep, accelerator)
            if accelerator.is_main_process:
                shutil.copy(f"{lora_dir}/adapter_model.safetensors", f"{save_dir}/adapter_model.safetensors")
            #     peft_model.save_pretrained(
            #         save_dir, 
            #         state_dict=state_dict,
            #         safe_serialization=True,              # writes .safetensors
            #         is_main_process=True,
            #         save_function=accelerator.save        # avoids multi-rank write conflicts
            #     )
        # vLLM spawns its own distributed executors; only run on main process
        # if accelerator.is_main_process:
        # TODO: maybe fix or remove and only rely on watcher.
        #     eval_hypermod_checkpoint(best_cp_path_hypermod, peft_adapter_path, accelerator.device, curstep, full_eval=True)
    elif ("mt_lora" in args.exp_setup) or ("mt_vera" in args.exp_setup):
        lora_dir = save_lora_checkpoint(save_dir, accelerator.unwrap_model(model), args.model_dir, curstep, accelerator)
        if accelerator.is_main_process:
            if not os.path.isfile(f"{save_dir}/adapter_model.safetensors"):
                shutil.copy(f"{lora_dir}/adapter_model.safetensors", f"{save_dir}/adapter_model.safetensors")
            if not os.path.isfile(f"{save_dir}/config.json"):
                shutil.copy(f"{lora_dir}/config.json", f"{save_dir}/config.json")
        #eval_lora(args, save_dir, curstep, full_eval=True)
    elif "mt_fullfinetune" in args.exp_setup:
        if accelerator.is_main_process:
            accelerator.unwrap_model(model).save_pretrained(save_dir)
        accelerator.wait_for_everyone()
    else:
        lora_dir = save_lora_checkpoint(save_dir, accelerator.unwrap_model(model), args.model_dir, curstep, accelerator)
        if accelerator.is_main_process:
            shutil.copy(f"{lora_dir}/adapter_model.safetensors", f"{save_dir}/adapter_model.safetensors")
            accelerator.unwrap_model(model).config.save_pretrained(save_dir)
        accelerator.wait_for_everyone()
        #eval_lora(args, save_dir, curstep, full_eval=True)

    if args.keep_only_best:
        if accelerator.is_main_process:
            # also keep the last checkpoint
            cp_dirs = sorted(glob(f"{save_dir}/checkpoints/it_*"), key=os.path.getmtime)
            for cp_dir in cp_dirs[:-1]:
                shutil.rmtree(cp_dir)
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        wandb.unwatch(hypermod)
    accelerator.end_training()
    neftune_hook_handle.remove()
    model.eval()
    if args.use_hypernet:
        hypermod.eval()

def validate(model, hypermod, val_dataloaders, _get_loss_batch, curstep, skip_benchmark=True):
    # skip_benchmark useful when only watcher.py is doing eval over benchmark.
    with torch.no_grad(), evaluating(model, hypermod):
        out = dict()
        for val_dataloader_name, val_dataloader in val_dataloaders.items():
            if skip_benchmark and ("benchmark" in val_dataloader_name):
                continue
            if val_dataloader is None:
                continue
            val_info = defaultdict(list)
            for val_batch in val_dataloader:
                if val_batch is None:
                    break
                batch_loss = _get_loss_batch(val_batch, return_per_token_acc=True, return_entropy=True)
                val_info["sft_loss"].append(batch_loss["sft_loss"].item())
                val_info["per_token_acc"].append(batch_loss["per_token_acc"].item())
                val_info["entropy"].append(batch_loss["entropy"].item())
            for k, v in val_info.items():
                val_info[k] = np.mean(v)
                log_scalar(f"{val_dataloader_name}/{k}", val_info[k], curstep)
            out[val_dataloader_name] = val_info
    return out


def save_lora_checkpoint(save_dir, model, model_dir, curstep, accelerator):
    lora_dir = f"{save_dir}/checkpoints/it_{curstep}/"
    
    # Only save on main process
    if accelerator.is_main_process:
        save_lora_from_peft_model(model, model_dir, lora_dir)
        if os.path.exists(f"{save_dir}/adapter_config.json"):
            shutil.copy(f"{save_dir}/adapter_config.json", f"{lora_dir}/adapter_config.json")
    accelerator.wait_for_everyone()
    
    return lora_dir


def log_train_vals(grad_norm, avg_losses, curstep, len_train_dataloader, scheduler, accelerator):
    accelerator.log(
        {
            "train/total_loss": np.mean(avg_losses["train/total_loss"]),
            "train/sft_loss": np.mean(avg_losses["train/sft_loss"]),
            "train/generated_w_l2_loss": np.mean(avg_losses["train/generated_w_l2_loss"]),
            "train/learning_rate": scheduler.get_last_lr()[0],
            "train/epoch": curstep / len_train_dataloader,
            "train/global_step": curstep,
            "train/grad_norm": grad_norm,
        },
        step=curstep,
    )
    logger.info(
        f"train/total_loss: {np.mean(avg_losses['train/total_loss']):.4f} "
        f"|| train/sft_loss: {np.mean(avg_losses['train/sft_loss']):.4f} "
        f"|| train/generated_w_l2_loss: {np.mean(avg_losses['train/generated_w_l2_loss']):.4f} "
    )


def compute_loss(labels, logits, equally_weight_sample, label_smoothing):
    bs = logits.shape[0]
    vocab_size = logits.shape[-1]
    # based on HG Transformers
    # modified to weight each example equally
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    max_seq_len = shift_labels.shape[1]
    seq_len = torch.where(shift_labels != -100, 1, 0).sum(-1, keepdim=True)
    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Ensure tensors are on the same device
    if equally_weight_sample:
        # weight each sample equally
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", label_smoothing=label_smoothing)
        loss = loss_fct(shift_logits, shift_labels)
        loss = (loss.view(bs, max_seq_len) / seq_len).sum(-1).mean()
    else:
        # weight each token equally
        loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        loss = loss_fct(shift_logits, shift_labels)
    return loss


def generate_and_hook_delta_w(
    target_modules,
    inp_dropout,
    model,
    layer_indices,
    hypermod,
    encoded_task_emb,
    bs,
    training,
):
    hook_handles = []
    factorized_delta_w = dict()
    # Unwrap DDP-wrapped model if needed for attribute access
    for target_module in target_modules:
        factorized_delta_w[target_module] = hypermod.get_delta_weights(
            layer_indices.repeat_interleave(bs),
            target_module,
            model.module if hasattr(model, "module") else model,
            encoded_task_emb.tile(layer_indices.shape[0], 1),
            factorized=True
        ) # [832, 8, 8] = [bs * L, r, r]
        # if not "z" in hypermod.exp_setup:
        lora_A, lora_B = factorized_delta_w[target_module]
        # print(lora_A.shape)
        for layer_index in layer_indices:
            start_indices, end_indices = layer_index * bs, (layer_index + 1) * bs
            handles = add_lora_hooks(
                model = model,
                module_names=[target_module],
                layer_indices=[layer_index],
                A=lora_A[start_indices:end_indices].transpose(-1, -2),  # [bs, in_features, r]
                B=lora_B[start_indices:end_indices].transpose(-1, -2),  # [bs, r, out_features]
                # Z=None,
                scaling=hypermod.scaling,
                input_dropout=inp_dropout,
                training=training,
                # z_type=hypermod.z_type
            )
            hook_handles += handles
    return factorized_delta_w, hook_handles
