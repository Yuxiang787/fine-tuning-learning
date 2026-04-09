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
    dtype: Optional[torch.dtype] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> Tuple[PreTrainedModel, torch.device]:
    """
    加载模型

    Args:
        model_name: 模型名称或路径
        device: 设备
        trust_remote_code: 是否信任远程代码
        dtype: 数据类型
        load_in_8bit: 是否 8 位量化加载
        load_in_4bit: 是否 4 位量化加载

    Returns:
        (模型，设备)
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    if dtype is None:
        if device.type == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32

    if device.type == "mps" and (load_in_8bit or load_in_4bit):
        raise ValueError("MPS 设备暂不支持 bitsandbytes 4-bit/8-bit 量化加载，请关闭 load_in_4bit/load_in_8bit。")

    # 量化加载 (QLoRA)
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
        # transformers >= 4.48.0 supports device_map="auto" on MPS
        use_device_map = device.type in ["cuda", "mps"]
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            device_map="auto" if use_device_map else None,
        )

        if not use_device_map:
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
    device = device or (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # 加载分词器
    tokenizer = load_tokenizer(base_model_name)

    # 加载基础模型
    base_model, device = load_model(base_model_name, device)

    # 加载适配器
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)

    return peft_model, tokenizer, device
