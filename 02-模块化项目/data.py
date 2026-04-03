#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载与处理模块
"""

import json
from typing import List, Dict, Any
from datasets import Dataset

from config import DataConfig


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    加载 JSONL 格式数据

    Args:
        path: 文件路径

    Returns:
        字典列表
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_dataset_from_file(path: str) -> Dataset:
    """
    从文件加载数据集

    Args:
        path: 数据文件路径

    Returns:
        HuggingFace Dataset
    """
    data = load_jsonl(path)
    return Dataset.from_list(data)


def format_example(
    example: Dict[str, Any],
    config: DataConfig
) -> Dict[str, str]:
    """
    将单个样本格式化为训练文本

    Args:
        example: 原始样本
        config: 数据配置

    Returns:
        包含 formatted_text 的新字典
    """
    instruction = example.get(config.data.instruction_column, "")
    input_text = example.get(config.data.input_column, "")
    output_text = example.get(config.data.output_column, "")

    # 根据是否有 input 选择模板
    if input_text:
        text = config.prompt_template.format(
            instruction=instruction,
            input=input_text,
            output=output_text
        )
    else:
        text = config.prompt_template_no_input.format(
            instruction=instruction,
            output=output_text
        )

    return {"text": text}


def create_training_dataset(
    data_path: str,
    config: DataConfig,
    tokenizer,
    max_length: int = 256
) -> Dataset:
    """
    创建训练数据集

    Args:
        data_path: 数据文件路径
        config: 数据配置
        tokenizer: 分词器
        max_length: 最大序列长度

    Returns:
        处理后的 Dataset
    """
    # 加载原始数据
    dataset = load_dataset_from_file(data_path)

    # 格式化
    dataset = dataset.map(
        lambda x: format_example(x, config),
        remove_columns=dataset.column_names
    )

    # 分词
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )

    tokenized_dataset = dataset.map(tokenize, batched=True)

    return tokenized_dataset


def preprocess_for_sft(
    dataset: Dataset,
    tokenizer,
    config: DataConfig,
    max_length: int = 256
) -> Dataset:
    """
    为 SFT（监督微调）预处理数据

    与 create_training_dataset 的区别：
    - 此函数会对 label 进行掩码处理，只计算 output 部分的 loss
    - 适合更精细的训练控制

    Args:
        dataset: 原始 Dataset
        tokenizer: 分词器
        config: 数据配置
        max_length: 最大长度

    Returns:
        处理后的 Dataset（包含 input_ids, attention_mask, labels）
    """
    def format_and_tokenize(example):
        # 构建完整文本
        if example.get(config.input_column):
            prompt = config.prompt_template.format(
                instruction=example[config.instruction_column],
                input=example[config.input_column],
                output=""  # 先留空
            )
        else:
            prompt = config.prompt_template_no_input.format(
                instruction=example[config.instruction_column],
                output=""
            )

        # 只对 prompt 分词
        prompt_tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            return_tensors=None
        )

        # 构建完整文本（含 output）用于训练
        if example.get(config.input_column):
            full_text = config.prompt_template.format(
                instruction=example[config.instruction_column],
                input=example[config.input_column],
                output=example[config.output_column]
            )
        else:
            full_text = config.prompt_template_no_input.format(
                instruction=example[config.instruction_column],
                output=example[config.output_column]
            )

        full_tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            return_tensors=None
        )

        # 创建 labels：prompt 部分用 -100 掩码，output 部分保留
        labels = [-100] * len(prompt_tokens["input_ids"]) + \
                 full_tokens["input_ids"][len(prompt_tokens["input_ids"]):]

        # 截断到 max_length
        input_ids = full_tokens["input_ids"][:max_length]
        attention_mask = full_tokens["attention_mask"][:max_length]
        labels = labels[:max_length]

        # padding
        padding_length = max_length - len(input_ids)
        if padding_length > 0:
            input_ids += [tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            labels += [-100] * padding_length

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    return dataset.map(format_and_tokenize, remove_columns=dataset.column_names)


def print_dataset_stats(dataset: Dataset, name: str = "Dataset"):
    """打印数据集统计信息"""
    print(f"\n{name} 统计信息:")
    print(f"  样本数量：{len(dataset)}")
    if len(dataset) > 0:
        print(f"  特征：{dataset.features}")
        print(f"  示例：{dataset[0]}")
