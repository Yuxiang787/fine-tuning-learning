#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型加载模块
"""

import torch
from typing import Tuple, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import PeftModel, PeftConfig


def load_tokenizer(
    model_name: str,
    trust_remote_code: bool = True,
    use_fast: bool = True
) -> PreTrainedTokenizer:
    """
    加载分词器

    Args:
        model_name: 模型名称或路径
        trust_remote_code: 是否信任远程代码
        use_fast: 是否使用快速分词器

    Returns:
        分词器
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast,
    )

    # 设置 padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def load_model(
    model_name: str,
    device: Optional[torch.device] = None,
    trust_remote_code: bool = True,
    torch_dtype: Optional[torch.dtype] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> Tuple[PreTrainedModel, torch.device]:
    """
    加载模型

    Args:
        model_name: 模型名称或路径
        device: 设备
        trust_remote_code: 是否信任远程代码
        torch_dtype: 数据类型
        load_in_8bit: 是否 8 位量化加载
        load_in_4bit: 是否 4 位量化加载

    Returns:
        (模型，设备)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch_dtype is None:
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    # 量化加载
    if load_in_8bit or load_in_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            quantization_config=quantization_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map="auto" if device.type == "cuda" else None,
        )

        if device.type != "cuda":
            model = model.to(device)

    return model, device


def load_peft_model(
    base_model_name: str,
    adapter_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[PeftModel, PreTrainedTokenizer, torch.device]:
    """
    加载 PEFT/LoRA 模型

    Args:
        base_model_name: 基础模型名称
        adapter_path: 适配器路径
        device: 设备

    Returns:
        (PEFT 模型，分词器，设备)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载分词器
    tokenizer = load_tokenizer(base_model_name)

    # 加载基础模型
    base_model, device = load_model(base_model_name, device)

    # 加载适配器
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)

    return peft_model, tokenizer, device
