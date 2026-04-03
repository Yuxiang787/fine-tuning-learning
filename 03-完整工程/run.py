#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主运行脚本 - 大模型微调训练

用法:
    python run.py --model_name Qwen/Qwen2.5-0.5B --data data.jsonl
    python run.py --config configs/lora_config.yaml
    python run.py --full_finetune
"""

import sys
import yaml
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.train.args import parse_args
from src.train.trainer import train


def load_config_from_yaml(path: str) -> dict:
    """从 YAML 文件加载配置"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    # 解析命令行参数
    args = parse_args()

    # 加载配置文件（如果指定）
    config = {}
    if args.config:
        print(f"从配置文件加载：{args.config}")
        config = load_config_from_yaml(args.config)

    # 命令行参数覆盖配置文件
    args_dict = vars(args)
    for key, value in args_dict.items():
        if value is not None:
            # 命令行参数优先级更高
            if key == 'config':
                continue
            config[key] = value

    # 执行训练
    train(config)


if __name__ == "__main__":
    main()
