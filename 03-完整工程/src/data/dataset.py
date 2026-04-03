#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集模块 - 加载和处理训练数据
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from datasets import Dataset


class FineTuningDataset:
    """微调数据集类"""

    def __init__(
        self,
        data_path: str,
        format_template: str,
        format_template_no_input: str,
        transform: Optional[Callable] = None
    ):
        """
        初始化数据集

        Args:
            data_path: 数据文件路径（JSONL）
            format_template: 有 input 时的格式化模板
            format_template_no_input: 无 input 时的格式化模板
            transform: 可选的转换函数（如分词）
        """
        self.data_path = Path(data_path)
        self.format_template = format_template
        self.format_template_no_input = format_template_no_input
        self.transform = transform

        self._data: Optional[Dataset] = None
        self._processed: Optional[Dataset] = None

    def load(self) -> "FineTuningDataset":
        """加载数据"""
        data = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        self._data = Dataset.from_list(data)
        return self

    def format(self) -> "FineTuningDataset":
        """格式化数据"""
        if self._data is None:
            raise ValueError("请先调用 load() 加载数据")

        def _format_sample(example):
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            output_text = example.get("output", "")

            if input_text:
                text = self.format_template.format(
                    instruction=instruction,
                    input=input_text,
                    output=output_text
                )
            else:
                text = self.format_template_no_input.format(
                    instruction=instruction,
                    output=output_text
                )

            return {"text": text}

        self._processed = self._data.map(_format_sample)
        return self

    def tokenize(self, tokenizer, max_length: int = 256) -> "FineTuningDataset":
        """分词处理"""
        if self._processed is None:
            raise ValueError("请先调用 format() 格式化数据")

        def _tokenize(example):
            return tokenizer(
                example["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

        self._processed = self._processed.map(_tokenize, batched=True)
        return self

    def to_huggingface(self) -> Dataset:
        """返回 HuggingFace Dataset"""
        if self._processed is None:
            raise ValueError("数据尚未处理")
        return self._processed

    def __len__(self) -> int:
        if self._data is None:
            return 0
        return len(self._data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        if self._data is None:
            raise ValueError("数据尚未加载")
        return self._data[idx]

    @property
    def raw(self) -> Optional[Dataset]:
        """返回原始数据"""
        return self._data

    @property
    def processed(self) -> Optional[Dataset]:
        """返回处理后的数据"""
        return self._processed


def load_dataset(
    data_path: str,
    format_template: str,
    format_template_no_input: str,
    tokenizer=None,
    max_length: int = 256
) -> Dataset:
    """
    快速加载数据集的便捷函数

    Args:
        data_path: 数据文件路径
        format_template: 格式化模板（有 input）
        format_template_no_input: 格式化模板（无 input）
        tokenizer: 分词器（可选）
        max_length: 最大长度

    Returns:
        处理后的 HuggingFace Dataset
    """
    dataset = FineTuningDataset(
        data_path=data_path,
        format_template=format_template,
        format_template_no_input=format_template_no_input
    )

    dataset.load().format()

    if tokenizer is not None:
        dataset.tokenize(tokenizer, max_length)

    return dataset.to_huggingface()


def format_sample(
    sample: Dict[str, Any],
    format_template: str,
    format_template_no_input: str
) -> str:
    """
    格式化单个样本

    Args:
        sample: 样本字典
        format_template: 有 input 的模板
        format_template_no_input: 无 input 的模板

    Returns:
        格式化后的文本
    """
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output_text = sample.get("output", "")

    if input_text:
        return format_template.format(
            instruction=instruction,
            input=input_text,
            output=output_text
        )
    else:
        return format_template_no_input.format(
            instruction=instruction,
            output=output_text
        )
