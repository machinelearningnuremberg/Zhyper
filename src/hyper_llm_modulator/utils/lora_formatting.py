from glob import glob
import torch
import os
import json
import shutil

from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file, save_file
import yaml


# from https://github.com/vllm-project/vllm/issues/4915#issuecomment-2119989248
def replicate_lora_a_qkv(
    name: str, weight: "torch.Tensor"
) -> dict[str, "torch.Tensor"]:
    prefix, suffix = name.split("qkv_proj")
    res = {}
    for t in ["q_proj", "k_proj", "v_proj"]:
        name = f"{prefix}{t}{suffix}"
        res[name] = weight.clone()
    return res


def replicate_lora_a_gate_up(
    name: str, weight: "torch.Tensor"
) -> dict[str, "torch.Tensor"]:
    prefix, suffix = name.split("gate_up_proj")
    res = {}
    for t in ["gate_proj", "up_proj"]:
        name = f"{prefix}{t}{suffix}"
        res[name] = weight.clone()
    return res


def split_lora_b_qkv(name: str, weight: "torch.Tensor") -> dict[str, "torch.Tensor"]:
    size = weight.shape[0] // 3
    prefix, suffix = name.split("qkv_proj")
    res = {
        f"{prefix}{t}{suffix}": w
        for t, w in zip(["q_proj", "k_proj", "v_proj"], weight.split(size))
    }
    return res


def split_lora_b_gate_up(
    name: str, weight: "torch.Tensor"
) -> dict[str, "torch.Tensor"]:
    size = weight.shape[0] // 2
    prefix, suffix = name.split("gate_up_proj")
    res = {
        f"{prefix}{t}{suffix}": w
        for t, w in zip(["gate_proj", "up_proj"], weight.split(size))
    }
    return res


def convert_qkv_gate_up_lora_to_splits_vllm(adapter_folder_path: str) -> None:
    """save the new adapter dict"""

    adapter_bin_name = "adapter_model.safetensors"
    adapter_config_name = "adapter_config.json"

    lora = load_file(f"{adapter_folder_path}/{adapter_bin_name}")
    with open(f"{adapter_folder_path}/{adapter_config_name}", "r") as f:
        lora_config = json.load(f)

    if not (
        "qkv_proj" in lora_config["target_modules"]
        or "gate_up_proj" in lora_config["target_modules"]
    ):
        print("No `qkv_proj` or `gate_up_proj` in the target_modules")
        return

    # converting weights
    res = {}
    for k, v in lora.items():
        if "qkv_proj" in k and "lora_A" in k:
            res.update(replicate_lora_a_qkv(k, v))
        elif "qkv_proj" in k and "lora_B" in k:
            res.update(split_lora_b_qkv(k, v))
        elif "gate_up_proj" in k and "lora_A" in k:
            res.update(replicate_lora_a_gate_up(k, v))
        elif "gate_up_proj" in k and "lora_B" in k:
            res.update(split_lora_b_gate_up(k, v))
        else:
            res[k] = v

    # converting config
    temp = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"] + [
        t
        for t in lora_config["target_modules"]
        if t != "qkv_proj" and t != "gate_up_proj"
    ]
    lora_config["target_modules"] = temp

    # backup
    for file in [adapter_bin_name, adapter_config_name]:
        os.rename(
            f"{adapter_folder_path}/{file}", f"{adapter_folder_path}/{file}.phi.bak"
        )

    save_file(
        res, f"{adapter_folder_path}/{adapter_bin_name}", metadata={"format": "pt"}
    )
    with open(f"{adapter_folder_path}/{adapter_config_name}", "w") as f:
        json.dump(lora_config, f, indent=4)


def lora_state_dict_to_tensor_dict(lora_sd, target_modules, layer_indices, device):
    A, B = (
        {
            target_module: [None for _ in range(len(layer_indices))]
            for target_module in target_modules
        },
        {
            target_module: [None for _ in range(len(layer_indices))]
            for target_module in target_modules
        },
    )

    for k, v in lora_sd.items():
        for target_module in target_modules:
            if target_module in k:
                layer_idx = int(k.split("layers.")[-1].split(".")[0])
                if layer_idx in layer_indices:
                    if "lora_A" in k:
                        A[target_module][layer_idx] = v.to(device)
                    elif "lora_B" in k:
                        B[target_module][layer_idx] = v.to(device)

    for target_module in target_modules:
        A[target_module] = torch.stack(A[target_module], dim=0)
        B[target_module] = torch.stack(B[target_module], dim=0)

    return dict(A=A, B=B)


def lora_tensor_dict_to_state_dict(
    lora_sd, module_names, target_modules, layer_indices
):
    # reverse what lora_state_dict_to_tensor_dict does
    lora_state_dict = dict()
    for target_module in target_modules:
        for layer_idx in layer_indices:
            for module_name in module_names[target_module][layer_idx]:
                if "lora_A" in module_name:
                    lora_state_dict[module_name] = (
                        lora_sd["A"][target_module][layer_idx].cpu().contiguous()
                    )
                elif "lora_B" in module_name:
                    lora_state_dict[module_name] = (
                        lora_sd["B"][target_module][layer_idx].cpu().contiguous()
                    )
                else:
                    raise ValueError(f"Unexpected module name: {module_name}")
    return lora_state_dict


def get_std_lora(lora_state_dicts):
    modules_names = lora_state_dicts[0].keys()
    std_lora = dict.fromkeys(modules_names)
    for module_name in modules_names:
        std_lora[module_name] = torch.stack(
            [lora_sd[module_name] for lora_sd in lora_state_dicts], dim=0
        )
        std_lora[module_name] = torch.std(std_lora[module_name], dim=0)
    return std_lora


def get_mean_lora(lora_state_dicts):
    modules_names = lora_state_dicts[0].keys()
    avg_lora = dict.fromkeys(modules_names)
    for module_name in modules_names:
        avg_lora[module_name] = torch.stack(
            [lora_sd[module_name] for lora_sd in lora_state_dicts], dim=0
        )
        avg_lora[module_name] = torch.mean(avg_lora[module_name], dim=0)
    return avg_lora


def get_target_lora_dirs(datasets, base_model_dir):
    oracle_lora_dir = "train_outputs/sft/oracle_lora"
    lora_dirs = glob(f"{oracle_lora_dir}/*/")
    out = dict()
    for d in lora_dirs:
        if not (
            os.path.exists(f"{d}/args.yaml")
            and os.path.exists(f"{d}/adapter_model.safetensors")
        ):
            continue
        lora_args = yaml.safe_load(open(f"{d}/args.yaml"))
        if base_model_dir.strip("/") != lora_args["model_dir"].strip("/"):
            continue

        # backward compatibility for old config names
        train_datasets = (
            lora_args["train_ds_names"]
            if "train_ds_names" in lora_args
            else lora_args["datasets"]
        )
        assert len(train_datasets) == 1, (
            "An oracle lora should be trained with only one task"
        )
        if train_datasets[0] in datasets:
            out[train_datasets[0]] = d
    if len(out) != len(datasets):
        missing_datasets = set(datasets) - set(out.keys())
        raise ValueError(
            f"Missing oracle lora directories for datasets: {missing_datasets}"
        )
    return out


def get_peft_module_names(model, target_modules, layer_indices):
    module_names = {
        target_module: [[] for _ in range(len(layer_indices))]
        for target_module in target_modules
    }
    for k in get_peft_model_state_dict(model):
        if ("lora" not in k) and ("vera_lambda" not in k):
            continue
        layer_idx = int(k.split("layers.")[-1].split(".")[0])
        if layer_idx in layer_indices:
            for target_module in target_modules:
                if target_module in k:
                    # if "vera_lambda" in k:
                    #     # replace the name to match the lora naming convention
                    #     k = k.replace("vera_lambda_d", "lora_A.weight")
                    #     k = k.replace("vera_lambda_b", "lora_B.weight")
                    module_names[target_module][layer_idx].append(k)
                    break
    return module_names


def convert_vera_to_lora(adapter_folder_path: str) -> None:
    """Convert VeRA adapter tensor keys to standard LoRA naming for vLLM.

    - Renames *.vera_lambda_d -> *.lora_A.weight
      and   *.vera_lambda_b -> *.lora_B.weight
    - Updates adapter_config.json peft_type to "LORA" if present
    - Ensures lora_alpha exists (defaults to 1) to satisfy loaders

    No-op if no VeRA keys are found.
    """
    adapter_bin_name = "adapter_model.safetensors"
    adapter_config_name = "adapter_config.json"
    adapter_bin_path = f"{adapter_folder_path}/{adapter_bin_name}"
    adapter_cfg_path = f"{adapter_folder_path}/{adapter_config_name}"

    if not (os.path.exists(adapter_bin_path) and os.path.exists(adapter_cfg_path)):
        return

    tensors = load_file(adapter_bin_path)
    has_vera = any(("vera_lambda_d" in k) or ("vera_lambda_b" in k) for k in tensors.keys())
    if not has_vera:
        return

    converted = {}
    vera_A = tensors.get("base_model.vera_A")
    vera_B = tensors.get("base_model.vera_B")

    for k, v in tensors.items():
        # Skip raw VeRA base matrices if we can materialize LoRA weights
        if k in ("base_model.vera_A", "base_model.vera_B"):
            continue
        if "vera_lambda_d" in k and vera_A is not None:
            # lora_A.weight = diag(lambda_d) @ A  -> broadcast as lambda_d[:, None] * A
            lambda_d = v
            lora_A = lambda_d.view(-1, 1) * vera_A
            new_k = k.replace("vera_lambda_d", "lora_A.weight")
            converted[new_k] = lora_A
        elif "vera_lambda_b" in k and vera_B is not None:
            # lora_B.weight = diag(lambda_b) @ B  -> broadcast as lambda_b[:, None] * B
            lambda_b = v
            # Match out-dim rows with lambda_b length if needed
            B = vera_B
            if B.shape[0] != lambda_b.shape[0]:
                B = B[: lambda_b.shape[0], :]
            lora_B = lambda_b.view(-1, 1) * B
            new_k = k.replace("vera_lambda_b", "lora_B.weight")
            converted[new_k] = lora_B
        elif ("vera_lambda_d" in k or "vera_lambda_b" in k) and (
            vera_A is None or vera_B is None
        ):
            # Fallback: rename only if base matrices missing (non-ideal)
            new_k = (
                k.replace("vera_lambda_d", "lora_A.weight")
                .replace("vera_lambda_b", "lora_B.weight")
            )
            converted[new_k] = v
        else:
            converted[k] = v

    with open(adapter_cfg_path, "r") as f:
        cfg = json.load(f)
    # Normalize peft_type and minimal fields expected by vLLM
    if "peft_type" in cfg and cfg["peft_type"].upper() != "LORA":
        cfg["peft_type"] = "LORA"
    cfg.setdefault("lora_alpha", 1)
    # Write back
    save_file(converted, adapter_bin_path, metadata={"format": "pt"})
    with open(adapter_cfg_path, "w") as f:
        json.dump(cfg, f, indent=4)


def save_lora_from_peft_model(model, model_dir, save_dir):
    model.save_pretrained(save_dir)
    model.base_model.config.save_pretrained(save_dir)
    if "Phi-3" in model_dir:
        convert_qkv_gate_up_lora_to_splits_vllm(save_dir)


def construct_full_lora_matrix(state_dict, target_modules, layer_indices, device):
    lora = lora_state_dict_to_tensor_dict(
        state_dict, target_modules, layer_indices, device
    )
    full_lora = dict()
    for module in target_modules:
        full_lora[module] = torch.bmm(lora["B"][module], lora["A"][module])
    return full_lora
