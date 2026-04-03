#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据收集器模块
"""

import torch
from typing import Dict, List, Any


class DataCollator:
    """数据收集器 - 将样本批量化"""

    def __init__(self, tokenizer, mlm: bool = False):
        """
        初始化

        Args:
            tokenizer: 分词器
            mlm: 是否使用掩码语言建模（False 为因果语言建模）
        """
        self.tokenizer = tokenizer
        self.mlm = mlm

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        将样本列表批量化

        Args:
            features: 样本列表

        Returns:
            批量化的张量字典
        """
        # 提取 input_ids 和 attention_mask
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]

        # 获取最大长度
        max_length = max(len(ids) for ids in input_ids)

        # padding
        input_ids_padded = []
        attention_mask_padded = []

        for ids, mask in zip(input_ids, attention_mask):
            padding_length = max_length - len(ids)
            input_ids_padded.append(
                ids + [self.tokenizer.pad_token_id] * padding_length
            )
            attention_mask_padded.append(
                mask + [0] * padding_length
            )

        # 转换为 tensor
        batch = {
            "input_ids": torch.tensor(input_ids_padded, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_padded, dtype=torch.long),
        }

        # 如果有 labels
        if "labels" in features[0]:
            labels = [f["labels"] for f in features]
            labels_padded = []
            for lbl in labels:
                padding_length = max_length - len(lbl)
                labels_padded.append(lbl + [-100] * padding_length)
            batch["labels"] = torch.tensor(labels_padded, dtype=torch.long)

        return batch
