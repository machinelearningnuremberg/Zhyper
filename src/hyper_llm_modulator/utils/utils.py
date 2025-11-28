import logging
from math import sqrt
from typing import Literal

import torch
from tqdm import tqdm
import wandb

logger = logging.getLogger("")


def embed_texts(texts, emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn, device, batch_size=None):
    formatted_descs = list(map(task_desc_format_fn, texts))
    tokenized_ds_descs = emb_tokenizer(
        formatted_descs,
        truncation=True,
        padding=True,
        max_length=2**13,
        return_tensors="pt",
    )
    return embed_tokens(tokenized_ds_descs, emb_model, pooling_fn, device, batch_size)


def embed_tokens(tokenized_texts, emb_model, pooling_fn, device, batch_size=None):
    if batch_size is None:
        # Process all at once if no batch size specified
        tokenized_texts = {k: v.to(device) for k, v in tokenized_texts.items()}
        return _embed_tokens_single_batch(tokenized_texts, emb_model, pooling_fn)

    # Process in batches
    n_samples = tokenized_texts["input_ids"].shape[0]
    embeddings = []

    for start_idx in tqdm(range(0, n_samples, batch_size), total=n_samples // batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch = {k: v[start_idx:end_idx].to(device) for k, v in tokenized_texts.items()}
        batch_embeddings = _embed_tokens_single_batch(batch, emb_model, pooling_fn)
        embeddings.append(batch_embeddings)

    return torch.cat(embeddings, dim=0)


def _embed_tokens_single_batch(tokenized_texts, emb_model, pooling_fn):
    outputs = emb_model(**tokenized_texts, output_hidden_states=True)
    task_embs = pooling_fn(outputs, tokenized_texts["attention_mask"]).to(torch.float32)
    return torch.nn.functional.normalize(task_embs) * sqrt(task_embs.shape[-1])


def get_inp_tokenize_fn(
    tokenizer,
    sft_mode: Literal["causal_lm", "completion"],
    is_intx_model: bool,
    inp_max_len: int,
):
    def tokenize_causal_lm(examples):
        # a dict with keys: ["input_ids", "attention_mask"]
        tokenized_seq = tokenizer(
            examples["text"],
            # apply_chat_template should already add all the special tokens
            add_special_tokens=True if not is_intx_model else False,
            truncation=True,
            padding=False,
            max_length=inp_max_len,
        )
        tokenized_seq["labels"] = tokenized_seq["input_ids"]
        return tokenized_seq

    # NOTE: we're not considering multi-turn sft
    # this fn is used to mask out the loss from the prompt
    # and train only on the response
    # see # see https://github.com/huggingface/trl/issues/632#issuecomment-1972630547
    # https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb
    # for more advanced multi-turn training
    def tokenize_prompt_completion(examples):
        # a dict with keys: ["input_ids", "attention_mask"]
        # we can also access seqeunce_ids to differentiate between prompt and response
        tokenized_seq = tokenizer(
            examples["prompt"],
            examples["response"],
            add_special_tokens=False,
            truncation=True,
            padding=False,
            # apply to prompt and response separately
            # i.e., we can get the max sequence length of 2 x inp_max_len
            max_length=inp_max_len,
        )

        tokenized_seq["labels"] = [None] * len(tokenized_seq["input_ids"])
        input_ids = tokenized_seq["input_ids"]
        attention_mask = tokenized_seq["attention_mask"]
        labels = tokenized_seq["labels"]
        for i in range(len(tokenized_seq["input_ids"])):
            if not is_intx_model:
                # manually add bos and eos tokens
                input_ids[i] = [tokenizer.bos_token_id] + input_ids[i] + [tokenizer.eos_token_id]
                attention_mask[i] = [1] + attention_mask[i] + [1]
                sequence_ids = [0] + tokenized_seq.sequence_ids(i) + [1]
            else:
                sequence_ids = tokenized_seq.sequence_ids(i)
            labels[i] = [-100 if sequence_id == 0 else label for sequence_id, label in zip(sequence_ids, input_ids[i])]
        return tokenized_seq

    tokenize_function = tokenize_causal_lm if sft_mode == "causal_lm" else tokenize_prompt_completion
    return tokenize_function


def log_scalar(metric_name, val, curstep):
    if wandb.run is not None:
        wandb.log({metric_name: val}, step=curstep)
    logger.info(f"{metric_name}: {val:.4f}")



#### LORA-XS UTILS 
#### https://github.com/MohammadrezaBanaei/LoRA-XS/blob/main/utils/initialization_utils.py

from sklearn.decomposition import TruncatedSVD
import numpy as np
from typing import Tuple
import torch.nn.functional as F

def run_svd(input_matrix: np.ndarray, rank: int, n_iter: int) -> Tuple[np.ndarray, TruncatedSVD]:
    svd = TruncatedSVD(n_components=rank, n_iter=n_iter)
    svd.fit(input_matrix)
    reduced_matrix = svd.transform(input_matrix)
    return reduced_matrix, svd

def get_linear_rec_svd(input_matrix: np.ndarray, rank: int, n_iter: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    reduced_matrix, svd = run_svd(input_matrix, rank, n_iter)

    reconstructed_matrix = svd.inverse_transform(reduced_matrix)
    return reconstructed_matrix, reduced_matrix, svd.components_


def replace_by_svd(weight, rank=8, n_iter=10):
    _, enc, dec = get_linear_rec_svd(weight.float().cpu().detach().numpy(), rank, n_iter)
    final_enc = torch.tensor(enc, dtype=weight.dtype, device=weight.device)
    final_dec = torch.tensor(dec, dtype=weight.dtype, device=weight.device)
    return final_enc, final_dec

def replace_module_weights(target_module: torch.nn.Module, new_weight: torch.Tensor, renormalize=False):
    device = target_module.weight.device
    dtype = target_module.weight.dtype

    new_weight = new_weight.to(device=device, dtype=dtype).contiguous()

    if renormalize:
        target_weight_norm = target_module.weight.norm()
        new_weight = new_weight * (target_weight_norm / (new_weight.norm() + 1e-8))

    with torch.no_grad():
        target_module.weight = torch.nn.Parameter(new_weight)

    for name, module in target_module.named_modules():
        if "lora_" in name:
            module.to(device)

def _transpose_lora_xs(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight

def init_module_weights(target_module: torch.nn.Linear, sigma: float):
    # Initialize weights with Gaussian distribution
    torch.nn.init.normal_(target_module.weight, mean=0, std=sigma)
    if hasattr(target_module, "bias"):
        # Set bias to zeros
        if target_module.bias is not None:
            torch.nn.init.zeros_(target_module.bias)


def get_delta_weight_lora_xs(self, adapter) -> torch.Tensor:
    # This function is introduced in newer PEFT versions. we modify this function instead of modifying
    # the merge function (as we did previously for version 0.4.0 of PEFT).
    """
    Compute the delta weight for the given adapter.

    Args:
        adapter (str):
            The name of the adapter for which the delta weight should be computed.
    """
    device = self.lora_B[adapter].weight.device
    dtype = self.lora_B[adapter].weight.dtype

    # In case users wants to merge the adapter weights that are in
    # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
    # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
    cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

    weight_A = self.lora_A[adapter].weight
    weight_B = self.lora_B[adapter].weight

    if cast_to_fp32:
        weight_A = weight_A.float()
        weight_B = weight_B.float()

    output_tensor = _transpose_lora_xs(
        weight_B @ self.default_lora_latent_mapping.weight @ weight_A,
        self.fan_in_fan_out
    ) * self.scaling[adapter]

    if cast_to_fp32:
        output_tensor = output_tensor.to(dtype=dtype)

        # cast back the weights
        self.lora_A[adapter].weight.data = weight_A.to(dtype)
        self.lora_B[adapter].weight.data = weight_B.to(dtype)

    return output_tensor


def forward_latent_lora_xs(self, x: torch.Tensor):
    previous_dtype = x.dtype

    if self.active_adapter[0] not in self.lora_A.keys():
        return F.linear(x, _transpose_lora_xs(self.weight, self.fan_in_fan_out), bias=self.bias)
    if self.disable_adapters:
        if self.r[self.active_adapter[0]] > 0 and self.merged:
            self.unmerge()
        result = F.linear(x, _transpose_lora_xs(self.weight, self.fan_in_fan_out), bias=self.bias)
    elif self.r[self.active_adapter[0]] > 0 and not self.merged:
        result = F.linear(x, _transpose_lora_xs(self.weight, self.fan_in_fan_out), bias=self.bias)

        x = x.to(self.lora_A[self.active_adapter[0]].weight.dtype)

        # adding latent_mapping in the forward loop
        result += (
            self.lora_B[self.active_adapter[0]](
                self.default_lora_latent_mapping(
                    self.lora_A[self.active_adapter[0]](self.lora_dropout[self.active_adapter[0]](x))
                )
            )
            * self.scaling[self.active_adapter[0]]
        )
    else:
        result = F.linear(x, _transpose_lora_xs(self.weight, self.fan_in_fan_out), bias=self.bias)

    result = result.to(previous_dtype)

    return result
    