#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练主脚本 - 统一入口

用法:
    python train.py              # 默认 LoRA 微调
    python train.py --full       # 全量微调
    python train.py --config config.json  # 使用配置文件
"""

import argparse
import json
from config import Config, default_config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="大模型微调训练脚本")

    parser.add_argument(
        "--full",
        action="store_true",
        help="使用全量微调（默认 LoRA）"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（JSON）"
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data.jsonl",
        help="训练数据路径"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="输出目录"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="模型名称或路径"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="训练轮数"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="批次大小"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="学习率"
    )

    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA 秩"
    )

    return parser.parse_args()


def load_config(args) -> Config:
    """加载配置"""
    if args.config:
        print(f"从文件加载配置：{args.config}")
        return Config.load(args.config)

    print("使用默认配置")
    config = Config()

    # 用命令行参数覆盖
    config.use_lora = not args.full
    config.model.model_name = args.model
    config.training.data_path = args.data
    config.training.output_dir = args.output
    config.training.num_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.lora.r = args.lora_r
    config.lora.alpha = args.lora_r * 2  # alpha 通常是 r 的 2 倍

    return config


def main():
    args = parse_args()
    config = load_config(args)

    print("\n" + "=" * 60)
    print("大模型微调训练")
    print("=" * 60)
    print(f"微调方式：{'全量微调' if not config.use_lora else 'LoRA'}")
    print(f"模型：{config.model.model_name}")
    print(f"数据：{config.training.data_path}")
    print(f"输出：{config.training.output_dir}")
    print(f"轮数：{config.training.num_epochs}")
    print(f"批次：{config.training.batch_size}")
    print(f"学习率：{config.training.learning_rate}")
    if config.use_lora:
        print(f"LoRA 秩：{config.lora.r}")
    print("=" * 60)

    if config.use_lora:
        from lora import train_lora
        train_lora(config)
    else:
        from full_finetune import train_full
        train_full(config)


if __name__ == "__main__":
    main()
