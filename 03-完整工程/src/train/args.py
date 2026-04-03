#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练参数解析
"""

import argparse
from pathlib import Path


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="大模型微调训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # LoRA 微调
  python run.py --model_name Qwen/Qwen2.5-0.5B --data data.jsonl

  # 全量微调
  python run.py --full_finetune --batch_size 2

  # 使用配置文件
  python run.py --config configs/lora_config.yaml
        """
    )

    # 模型参数
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="模型名称或路径"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="分词器名称（默认与 model_name 相同）"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="信任远程代码"
    )

    # 数据参数
    parser.add_argument(
        "--data",
        type=str,
        default="data.jsonl",
        help="训练数据路径"
    )
    parser.add_argument(
        "--validation_data",
        type=str,
        default=None,
        help="验证数据路径"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="最大序列长度"
    )

    # LoRA 参数
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=True,
        help="使用 LoRA（默认启用）"
    )
    parser.add_argument(
        "--full_finetune",
        action="store_true",
        help="全量微调（禁用 LoRA）"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA 秩"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha 参数"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout"
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=None,
        help="LoRA 目标模块"
    )

    # 训练参数
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="输出目录"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="训练轮数"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="批次大小"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="学习率"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="预热步数"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="梯度累积步数"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="权重衰减"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="最大梯度范数"
    )

    # 日志与保存
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="日志记录步数"
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        choices=["no", "epoch", "steps"],
        help="保存策略"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="保存步数间隔"
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        choices=["none", "wandb", "tensorboard"],
        help="报告工具"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="lora-qwen0.5b-m4",
        help="运行名称"
    )

    # 性能参数
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="使用 FP16 混合精度"
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="使用 BF16 混合精度"
    )
    parser.add_argument(
        "--dataloader_workers",
        type=int,
        default=0,
        help="数据加载工作进程数"
    )

    # 其他
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（YAML）"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从检查点恢复训练"
    )

    return parser.parse_args()
