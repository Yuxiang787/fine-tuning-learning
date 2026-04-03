#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块 - 统一管理所有超参数和路径
"""

from dataclasses import dataclass, field
from typing import Optional, List
import json


@dataclass
class ModelConfig:
    """模型配置"""
    # M4 24GB 推荐：Qwen/Qwen2.5-0.5B (中文友好) 或 TinyLlama/TinyLlama-1.1B-Chat-v1.0
    model_name: str = "Qwen/Qwen2.5-0.5B"
    trust_remote_code: bool = True
    use_flash_attention: bool = False


@dataclass
class LoraConfig:
    """LoRA 配置"""
    r: int = 16         # M4 24GB 推荐 16
    alpha: int = 32     # 2*r
    dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    bias: str = "none"


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础参数 - M4 24GB 推荐
    output_dir: str = "./output"
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-4
    warmup_steps: int = 10

    # 数据参数
    max_length: int = 512
    data_path: str = "data.jsonl"

    # 日志与保存
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_total_limit: int = 3

    # 性能优化
    gradient_accumulation_steps: int = 2
    fp16: bool = True
    dataloader_num_workers: int = 0

    # 其他
    seed: int = 42
    report_to: str = "none"  # 或 "wandb"


@dataclass
class DataConfig:
    """数据配置"""
    # 数据格式
    instruction_column: str = "instruction"
    input_column: str = "input"
    output_column: str = "output"

    # 提示模板
    prompt_template: str = """### Instruction:
{instruction}

### Input:
{input}

### Output:
{output}
"""

    # 无 input 时的模板
    prompt_template_no_input: str = """### Instruction:
{instruction}

### Output:
{output}
"""


@dataclass
class Config:
    """总配置类"""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # 是否使用 LoRA（False 则为全量微调）
    use_lora: bool = True

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "use_lora": self.use_lora,
            "model": self.model.__dict__,
            "lora": self.lora.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
        }

    def save(self, path: str):
        """保存配置到 JSON 文件"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Config":
        """从 JSON 文件加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = cls()
        config.use_lora = data.get("use_lora", True)

        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "lora" in data:
            config.lora = LoraConfig(**data["lora"])
        if "training" in data:
            config.training = TrainingConfig(**data["training"])
        if "data" in data:
            config.data = DataConfig(**data["data"])

        return config


# 默认配置实例
default_config = Config()
