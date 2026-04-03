#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA 工具模块
"""

from typing import Optional, List, Tuple
from transformers import PreTrainedModel
from peft import LoraConfig, get_peft_model, TaskType


def create_lora_config(
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    task_type: TaskType = TaskType.CAUSAL_LM,
) -> LoraConfig:
    """
    创建 LoRA 配置

    Args:
        r: LoRA 秩
        alpha: 缩放系数
        dropout: Dropout 比率
        target_modules: 目标模块列表（None 为自动选择）
        bias: 是否训练 bias
        task_type: 任务类型

    Returns:
        LoraConfig 实例
    """
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias=bias,
        task_type=task_type,
    )


def apply_lora(
    model: PreTrainedModel,
    lora_config: LoraConfig
) -> Tuple[PreTrainedModel, int, int]:
    """
    对模型应用 LoRA

    Args:
        model: 基础模型
        lora_config: LoRA 配置

    Returns:
        (LoRA 模型，可训练参数数量，总参数数量)
    """
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    return model, trainable_params, total_params


def print_lora_info(model: PreTrainedModel):
    """打印 LoRA 模型信息"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"\nLoRA 模型信息:")
    print(f"  总参数量：{total:,}")
    print(f"  可训练参数：{trainable:,} ({100*trainable/total:.2f}%)")

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
