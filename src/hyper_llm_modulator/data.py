from collections import defaultdict
from functools import partial
from glob import glob
from math import ceil
import hashlib
import json
import logging
import os
import random
from typing import Union

import torch
import datasets
from datasets import load_dataset
from torch.utils.data import Dataset, ConcatDataset, Sampler, DataLoader

from hyper_llm_modulator.utils import (
    embed_texts,
    get_inp_tokenize_fn,
    get_preprocessing_fn,
    get_prompt_formatting_fn,
    repeat_iterator,
)

logger = logging.getLogger()

DATA_DIR = "data"
TRANSFORMED_DS_DIR = "data/transformed_datasets"
EMBS_DIR = "data/embs"
BENCHMARK_TASK_INFO = {
    "openbookqa": {"split": "validation[:500]"},
    "hellaswag": {"split": "train[:500]"},
    "winogrande": {"name": "winogrande_debiased", "split": "train[:500]", "trust_remote_code": True},
    "boolq": {"split": "train[:500]"},
    "piqa": {"split": "train[:500]"},
    "arc_easy": {"name": "ARC-Easy", "split": "validation[:500]"},
    "arc_challenge": {"name": "ARC-Challenge", "split": "validation[:500]"},
}


class PerTaskEmbSFTDataset(Dataset):
    def __init__(self, tokenized_dataset: datasets.Dataset, task_embs: torch.Tensor, validation: bool):
        self.tokenized_dataset = tokenized_dataset
        self.task_embs = task_embs
        self.validation = validation

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        example = self.tokenized_dataset[idx]
        emb_idx = idx % len(self.task_embs) if self.validation else random.randint(0, len(self.task_embs) - 1)
        task_emb = self.task_embs[emb_idx]
        example["task_emb"] = task_emb
        return example


class PerSampleEmbSFTDataset(Dataset):
    def __init__(
        self,
        tokenized_dataset: datasets.Dataset,
        task_embs: torch.Tensor,
        validation: bool,
    ):
        assert len(tokenized_dataset) == len(task_embs)
        self.tokenized_dataset = tokenized_dataset
        self.task_embs = task_embs
        self.validation = validation

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        example = self.tokenized_dataset[idx]
        task_emb = self.task_embs[idx]
        example["task_emb"] = task_emb
        return example


class HierachicalBatchSampler(Sampler):
    # a sampler that first samples which dataset to sample from
    # then samples from that dataset
    # only works with ConcatDataset

    def __init__(self, concat_dataset: ConcatDataset, n_ds_per_batch: int, n_points_per_ds: int):
        self.concat_dataset = concat_dataset
        self.n_ds_per_batch = n_ds_per_batch
        self.n_points_per_ds = n_points_per_ds
        self.cumulative_sizes = concat_dataset.cumulative_sizes
        self.n_datasets = len(self.cumulative_sizes)
        self.ds_sizes = [len(ds) for ds in concat_dataset.datasets]
        self.batch_size = n_ds_per_batch * n_points_per_ds

    def __len__(self):
        return self.n_datasets // self.n_ds_per_batch

    def __iter__(self):
        # TODO: iterate over all samples in one epoch
        task_indices = torch.randperm(self.n_datasets)
        for i in range(0, self.n_datasets, self.n_ds_per_batch):
            batch_indices = []
            if i + self.n_ds_per_batch > self.n_datasets:
                # drop the last batch if it's too small
                break

            for j in range(i, i + self.n_ds_per_batch):
                ds_idx = task_indices[j]
                ds_size = self.ds_sizes[ds_idx]
                local_indices = torch.randint(0, ds_size, (self.n_points_per_ds,))
                global_indices = local_indices + self.cumulative_sizes[ds_idx] - ds_size
                batch_indices.extend(global_indices.tolist())

            yield batch_indices


def get_datasets(dataset_names, metadata, tokenizer, sft_mode, is_intx_model, inp_max_len):
    out = dict()
    dataset_info_dict = {k: metadata[k]["ds_kwargs"] for k in dataset_names}
    inp_tokenize_fn = get_inp_tokenize_fn(tokenizer, sft_mode, is_intx_model, inp_max_len)
    for i, (ds_name, ds_kwargs) in enumerate(dataset_info_dict.items()):
        logger.debug(f"ds_name: {ds_name}, ds_kwargs: {ds_kwargs}")
        # get hash for the dataset
        ds_repr = f"{ds_name}_{json.dumps(ds_kwargs)}_{tokenizer.name_or_path.strip('/')}_{sft_mode}_{is_intx_model}_{inp_max_len}"
        ds_repr += f"_{json.dumps(metadata[ds_name])}"
        ds_hash = hashlib.sha256(ds_repr.encode("utf-8")).hexdigest()
        if glob(f"{TRANSFORMED_DS_DIR}/{ds_hash}/"):
            logger.debug(f"Loading preprocessed dataset: {ds_hash}")
            tokenized_dataset = datasets.load_from_disk(f"{TRANSFORMED_DS_DIR}/{ds_hash}")
        else:
            formatted_dataset = load_and_format_dataset(
                metadata, tokenizer, sft_mode, is_intx_model, ds_name, ds_kwargs
            )
            logger.debug(f"formatted example: {formatted_dataset[:5]}")
            tokenized_dataset = formatted_dataset.map(
                inp_tokenize_fn, batched=True, remove_columns=formatted_dataset.column_names
            )
            logger.debug(f"tokenized example: {tokenized_dataset[:5]}")
            tokenized_dataset.set_format("torch")

            logger.debug(f"Saving preprocessed dataset: {ds_hash}")
            tokenized_dataset.save_to_disk(f"{TRANSFORMED_DS_DIR}/{ds_hash}")

        out[ds_name] = tokenized_dataset

    return out


def load_and_format_dataset(metadata, tokenizer, sft_mode, is_intx_model, ds_name, ds_kwargs):
    ds_repr = f"{ds_name}_{json.dumps(ds_kwargs)}_{tokenizer.name_or_path.strip('/')}_{sft_mode}_{is_intx_model}"
    ds_repr += f"_{json.dumps(metadata[ds_name])}"
    ds_hash = hashlib.sha256(ds_repr.encode("utf-8")).hexdigest()
    if glob(f"{TRANSFORMED_DS_DIR}/{ds_hash}/"):
        logger.debug(f"Loading preprocessed dataset: {ds_hash}")
        formatted_dataset = datasets.load_from_disk(f"{TRANSFORMED_DS_DIR}/{ds_hash}")
    else:
        if "trust_remote_code" not in list(ds_kwargs.keys()):
            ds_kwargs["trust_remote_code"] = True
        dataset = load_dataset(**ds_kwargs)
        processed_dataset = dataset.map(get_preprocessing_fn(ds_name), batched=False)

        prompt_formatting_fn = get_prompt_formatting_fn(
            metadata[ds_name], sft_mode, tokenizer.apply_chat_template, is_intx_model
        )
        formatted_dataset = processed_dataset.map(prompt_formatting_fn, batched=True)
        formatted_dataset.save_to_disk(f"{TRANSFORMED_DS_DIR}/{ds_hash}")
    return formatted_dataset


@torch.no_grad()
def get_task_embs(
    ds_descs,
    emb_model,
    emb_tokenizer,
    task_desc_format_fn,
    pooling_fn,
    device,
):
    out = dict()
    for i, (ds_name, descs) in enumerate(ds_descs.items()):
        task_embs = None
        # pre-embed task descriptions when using per-task descriptions
        if emb_model is not None:
            # NOTE: assume that the number of task descs are small so we pad them here only once
            # logger.debug(f"{ds_descs[ds_name]=}")
            task_embs = embed_texts(descs, emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn, device)
        else:
            # one-hot task indicator
            task_embs = torch.eye(len(ds_descs), device=device)[i].unsqueeze(0)

        # logger.debug(f"{task_embs=}")
        out[ds_name] = task_embs
    return out


def collator(inp_list, tokenizer):
    # input is a list of tokenized sequences
    padding_kwargs = dict(padding=True, pad_to_multiple_of=8, return_tensors="pt")
    labels = [x.pop("labels") for x in inp_list]
    task_embs = task_descs = None
    if "task_emb" in inp_list[0]:
        task_embs = torch.stack([x.pop("task_emb") for x in inp_list])

    padded_seq = tokenizer.pad(inp_list, **padding_kwargs)

    # hacky explicit padding since the labels are not padded by default
    labels = tokenizer.pad({"input_ids": labels}, **padding_kwargs)["input_ids"]
    labels = torch.where(padded_seq["attention_mask"] == 0, -100, labels)
    out = {**padded_seq, "labels": labels}
    if task_embs is not None:
        out["task_embs"] = task_embs
    return out


@torch.no_grad()
def get_dataloader(
    ds_dict,
    task_embs_dict,
    tokenizer,
    use_per_task_emb,
    use_inp_as_desc,
    use_per_sample_desc,
    n_tasks_per_batch,
    n_points_per_task,
    use_hierarchical_sampler,
    batch_size,  # only needed for random sampler
    validation,
):
    if task_embs_dict is not None:
        assert len(ds_dict) == len(task_embs_dict)

    ds_list = []
    for ds_name in ds_dict:
        if use_per_task_emb:
            ds_list.append(PerTaskEmbSFTDataset(ds_dict[ds_name], task_embs_dict[ds_name], validation))
        elif use_inp_as_desc or use_per_sample_desc:
            ds_list.append(PerSampleEmbSFTDataset(ds_dict[ds_name], task_embs_dict[ds_name], validation))
        else:
            # no-op
            ds_list.append(ds_dict[ds_name])

    dataset = torch.utils.data.ConcatDataset(ds_list)
    if use_hierarchical_sampler:
        sampler = HierachicalBatchSampler(dataset, n_tasks_per_batch, n_points_per_task)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)
        sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=False)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=partial(collator, tokenizer=tokenizer))


def create_dataloaders(
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
):

    _get_datasets = partial(
        get_datasets,
        tokenizer=tokenizer,
        sft_mode=args.sft_mode,
        is_intx_model=is_intx_model,
        inp_max_len=args.inp_max_len,
    )
    _get_dataloader = partial(
        get_dataloader,
        tokenizer=tokenizer,
        use_per_task_emb=args.use_per_task_emb,
        use_inp_as_desc=args.use_inp_as_desc,
        use_per_sample_desc=args.use_per_sample_desc,
        n_tasks_per_batch=args.n_tasks_per_batch,
        n_points_per_task=args.n_points_per_task,
    )

    if emb_model is not None:
        emb_model = emb_model.eval()

    val_ds_names = []
    benchmark_val_ds_names = []
    unseen_val_ds_names = []

    for ds_name in args.train_ds_names:
        # by default, we load max 10,000 samples for each task (see tasks/lol_*/metadata.yaml)
        # which, as far as i can tell, does not actually cap the number of samples
        if ds_name in args.eval_ds_info:
            # make a validation split for tasks that are in both training and validation
            train_metadata[ds_name]["ds_kwargs"]["split"] = "train[:90%]"
            if ds_name == "longreward":
                train_metadata[ds_name]["ds_kwargs"]["split"] = "sft[:90%]"

    val_ds_names = [name for name in args.eval_ds_info if name in args.train_ds_names]
    for ds_name in val_ds_names:
        val_metadata[ds_name]["ds_kwargs"]["split"] = "train[90%:]"
        if ds_name == "longreward":
            val_metadata[ds_name]["ds_kwargs"]["split"] = "sft[90%:]"

    if use_hypernet or "mt" in args.exp_setup:
        # meta-validation datasets
        unseen_val_ds_names = [
            name for name in args.eval_ds_info if name.startswith("lol") and name not in args.train_ds_names
        ]
        benchmark_val_ds_names = [t for t in BENCHMARK_TASK_INFO if t in val_metadata]
        for ds_name in unseen_val_ds_names:
            val_metadata[ds_name]["ds_kwargs"]["split"] = "valid[:500]"
        for ds_name in benchmark_val_ds_names:
            val_metadata[ds_name]["ds_kwargs"].update(BENCHMARK_TASK_INFO[ds_name])

    out = {"train": None, "val/seen": None, "val/unseen": None, "val/benchmark": None}
    ds_names_list = [args.train_ds_names, val_ds_names, unseen_val_ds_names, benchmark_val_ds_names]

    logging.info(f"{args.use_per_task_emb=}\n{args.use_inp_as_desc=}\n{args.use_per_sample_desc=}")

    for split_name, ds_names in zip(out, ds_names_list):
        logger.info(f"{split_name=}, {ds_names=}")
        if len(ds_names) == 0:
            continue
        metadata = train_metadata if split_name == "train" else val_metadata
        ds_dict = _get_datasets(ds_names, metadata)
        ds_embs_dict = get_embs_dict(
            args,
            emb_model,
            emb_tokenizer,
            task_desc_format_fn,
            pooling_fn,
            ds_names,
            metadata,
            device,
        )

        train_kwargs = dict(
            use_hierarchical_sampler=args.use_hierarchical_sampler,
            batch_size=args.batch_size,
            validation=False,
        )
        val_kwargs = dict(use_hierarchical_sampler=False, batch_size=args.val_batch_size, validation=True)
        kwargs = train_kwargs if split_name == "train" else val_kwargs

        out[split_name] = _get_dataloader(ds_dict, ds_embs_dict, **kwargs)

    return out


def get_embs_dict(args, emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn, ds_names, metadata, device):
    ds_descs_dict = {ds: metadata[ds]["descriptions"][: args.n_descs_per_ds] for ds in ds_names}
    _get_task_embs = partial(
        get_task_embs,
        emb_model=emb_model,
        emb_tokenizer=emb_tokenizer,
        task_desc_format_fn=task_desc_format_fn,
        pooling_fn=pooling_fn,
        device=device,
    )

    ds_embs_dict = None
    if args.use_per_task_emb:
        ds_embs_dict = _get_task_embs(ds_descs_dict)
    elif args.use_inp_as_desc or args.use_default_desc or args.use_per_sample_desc:
        # the description has to be tokenized by emb_tokenizer not the base model's tokenizer
        ds_descs_dict = {
            ds_name: load_and_format_dataset(
                metadata,
                emb_tokenizer,
                args.sft_mode,
                is_intx_model=emb_tokenizer.chat_template is not None,
                ds_name=ds_name,
                ds_kwargs=metadata[ds_name]["ds_kwargs"],
            )
            for ds_name in ds_names
        }
        if args.use_per_sample_desc:
            ds_descs_dict = {ds_name: ds_descs_dict[ds_name]["context"] for ds_name in ds_names}
        else:
            ds_descs_dict = {ds_name: ds_descs_dict[ds_name]["prompt"] for ds_name in ds_names}
            if args.use_default_desc:
                for ds_name in ds_descs_dict:
                    prompts = ds_descs_dict[ds_name]
                    for i in range(len(prompts)):
                        prompts[i] = prompts[i].split("\n\n")[0]
                    ds_descs_dict[ds_name] = prompts
        ds_embs_dict = get_inp_prompt_emb(
            emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn, ds_names, ds_descs_dict
        )
    elif args.use_one_hot_task_emb:
        ds_embs_dict = {
            ds_name: torch.eye(len(ds_names), device=device)[i].unsqueeze(0) for i, ds_name in enumerate(ds_names)
        }

    return ds_embs_dict


@torch.no_grad()
def get_inp_prompt_emb(emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn, ds_names, ds_descs_dict):
    ds_embs_dict = {}
    for ds_name in ds_names:
        ds_repr = f"{ds_name}_{json.dumps(ds_descs_dict[ds_name])}_{emb_tokenizer.name_or_path.strip('/')}"
        ds_hash = hashlib.sha256(ds_repr.encode("utf-8")).hexdigest()
        os.makedirs(f"{EMBS_DIR}/", exist_ok=True)
        if glob(f"{EMBS_DIR}/{ds_hash}.pt"):
            logger.debug(f"Loading preprocessed dataset: {ds_hash}")
            ds_embs_dict[ds_name] = torch.load(f"{EMBS_DIR}/{ds_hash}.pt", map_location="cpu")
        else:
            ds_embs_dict[ds_name] = embed_texts(
                ds_descs_dict[ds_name],
                emb_model,
                emb_tokenizer,
                task_desc_format_fn,
                pooling_fn,
                device=emb_model.device,
                batch_size=32,
            ).to("cpu")
            torch.save(ds_embs_dict[ds_name], f"{EMBS_DIR}/{ds_hash}.pt")
    return ds_embs_dict


@torch.no_grad()
def get_recon_train_data(state_dict, target_modules, layer_indices, device, output_delta_w=False):
    layer_indices_out, lora_A, lora_B, target_deltaW = (
        defaultdict(list),
        {target_module: [None for _ in range(len(layer_indices))] for target_module in target_modules},
        {target_module: [None for _ in range(len(layer_indices))] for target_module in target_modules},
        dict(),
    )

    for k, v in state_dict.items():
        for target_module in target_modules:
            if target_module in k:
                layer_idx = int(k.split("layers.")[-1].split(".")[0])
                if layer_idx in layer_indices:
                    if "lora_A" in k:
                        lora_A[target_module][layer_idx] = v
                        layer_indices_out[target_module].append(layer_idx)
                    elif "lora_B" in k:
                        lora_B[target_module][layer_idx] = v

    for target_module in target_modules:
        lora_A[target_module] = torch.stack(lora_A[target_module], dim=0).to(device)
        lora_B[target_module] = torch.stack(lora_B[target_module], dim=0).to(device)
        if output_delta_w:
            target_deltaW[target_module] = (
                torch.bmm(
                    lora_B[target_module],
                    lora_A[target_module],
                )
                .to(torch.float32)
                .to(device)
            )

        layer_indices_out[target_module] = torch.tensor(
            sorted(layer_indices_out[target_module]),
            dtype=torch.long,
            device=device,
        )

    return dict(
        layer_indices=layer_indices_out,
        lora_A=lora_A,
        lora_B=lora_B,
        target_deltaW=target_deltaW,
    )


if __name__ == "__main__":
    from datasets import load_dataset

    seed = 42
    ds1 = load_dataset("Lots-of-LoRAs/task022_cosmosqa_passage_inappropriate_binary", "default", split="train[:5]")
    ds2 = load_dataset("Lots-of-LoRAs/task033_winogrande_answer_generation", split="train[:5]")
    ds3 = load_dataset("Lots-of-LoRAs/task034_winogrande_question_modification_object", split="train[:5]")
    ds4 = load_dataset("Lots-of-LoRAs/task035_winogrande_question_modification_person", split="train[:5]")
    dataset = ConcatDataset([ds1, ds2, ds3, ds4])
    sampler = HierachicalBatchSampler(dataset, 2, 2)
    dataloader = DataLoader(dataset, batch_sampler=sampler)
    breakpoint()
    for batch in repeat_iterator(dataloader):
        print(batch["id"])
        breakpoint()
    print("done")
