#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型加载与配置模块
"""

import torch
from typing import Optional, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
)

from config import ModelConfig, LoraConfig as LoraConfigDataclass


def load_tokenizer(
    model_name: str,
    trust_remote_code: bool = True
) -> PreTrainedTokenizer:
    """
    加载分词器

    Args:
        model_name: 模型名称或路径
        trust_remote_code: 是否信任远程代码

    Returns:
        分词器
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )

    # 设置 padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def load_base_model(
    model_name: str,
    device: Optional[torch.device] = None,
    trust_remote_code: bool = True,
    use_flash_attention: bool = False
) -> PreTrainedModel:
    """
    加载基础模型

    Args:
        model_name: 模型名称或路径
        device: 设备
        trust_remote_code: 是否信任远程代码
        use_flash_attention: 是否使用 Flash Attention

    Returns:
        基础模型
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # 确定数据类型
    torch_dtype = torch.float16 if device.type in ["cuda", "mps"] else torch.float32

    # 加载模型
    # transformers >= 4.48.0 supports device_map="auto" on MPS
    use_device_map = device.type in ["cuda", "mps"]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        device_map="auto" if use_device_map else None,
    )

    # 如果不用 device_map="auto"，手动移动模型
    if not use_device_map:
        model = model.to(device)

    return model


def create_lora_config(
    lora_config: LoraConfigDataclass,
    task_type: TaskType = TaskType.CAUSAL_LM
) -> LoraConfig:
    """
    创建 LoRA 配置

    Args:
        lora_config: LoRA 配置数据类
        task_type: 任务类型

    Returns:
        PEFT LoRAConfig
    """
    return LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        target_modules=lora_config.target_modules,
        lora_dropout=lora_config.dropout,
        bias=lora_config.bias,
        task_type=task_type,
    )


def apply_lora(
    model: PreTrainedModel,
    lora_config: LoraConfig
) -> Tuple[PeftModel, int, int]:
    """
    对模型应用 LoRA

    Args:
        model: 基础模型
        lora_config: LoRA 配置

    Returns:
        (应用 LoRA 后的模型，可训练参数数量，总参数数量)
    """
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    return model, trainable_params, total_params


def print_model_info(
    model: PreTrainedModel,
    name: str = "Model"
):
    """打印模型信息"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"\n{name} 信息:")
    print(f"  总参数量：{total:,}")
    print(f"  可训练参数：{trainable:,} ({100*trainable/total:.2f}%)")

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()


def load_lora_adapter(
    base_model: PreTrainedModel,
    adapter_path: str
) -> PeftModel:
    """
    加载 LoRA 适配器到基础模型

    Args:
        base_model: 基础模型
        adapter_path: 适配器路径

    Returns:
        带有适配器的模型
    """
    return PeftModel.from_pretrained(base_model, adapter_path)


def merge_lora_weights(
    peft_model: PeftModel
) -> PreTrainedModel:
    """
    合并 LoRA 权重到基础模型

    Args:
        peft_model: PEFT 模型

    Returns:
        合并后的基础模型（可安全导出）
    """
    return peft_model.merge_and_unload()


def get_device() -> torch.device:
    """获取当前设备"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
