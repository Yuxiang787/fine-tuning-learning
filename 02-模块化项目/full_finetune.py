#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全量微调模块 - 微调所有模型参数
"""

import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

from config import Config
from model import (
    load_tokenizer,
    load_base_model,
    print_model_info,
    get_device,
)
from data import create_training_dataset


def create_full_finetune_trainer(
    model,
    tokenizer,
    train_dataset,
    config: Config
) -> Trainer:
    """
    创建全量微调的 Trainer

    与 LoRA 的区别：
    - 所有参数都可训练
    - 需要更多显存
    - 通常需要更小的学习率
    """

    # 全量微调通常需要更小的学习率
    learning_rate = config.training.learning_rate / 10  # 1e-5 量级

    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.batch_size,
        learning_rate=learning_rate,
        warmup_steps=config.training.warmup_steps,
        logging_steps=config.training.logging_steps,
        save_strategy=config.training.save_strategy,
        save_total_limit=config.training.save_total_limit,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        fp16=config.training.fp16 and get_device().type in ["cuda", "mps"],
        dataloader_num_workers=config.training.dataloader_num_workers,
        seed=config.training.seed,
        report_to=config.training.report_to,
        # 全量微调特有参数
        weight_decay=0.01,  # 权重衰减，防止过拟合
        max_grad_norm=1.0,  # 梯度裁剪
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )


def train_full(config: Config) -> tuple:
    """
    执行全量微调训练

    Args:
        config: 配置

    Returns:
        (训练好的模型，分词器)
    """
    device = get_device()
    print(f"使用设备：{device}")

    # 1. 加载分词器
    print(f"\n加载分词器：{config.model.model_name}")
    tokenizer = load_tokenizer(config.model.model_name)

    # 2. 加载基础模型
    print(f"加载模型：{config.model.model_name}")
    model = load_base_model(
        config.model.model_name,
        device=device,
        trust_remote_code=config.model.trust_remote_code,
    )

    # 3. 打印模型信息（全量微调时所有参数都可训练）
    print_model_info(model, "基础模型")
    print("全量微调：所有参数都可训练")

    # 4. 准备数据
    print(f"\n准备训练数据：{config.training.data_path}")
    train_dataset = create_training_dataset(
        config.training.data_path,
        config.data,
        tokenizer,
        config.training.max_length
    )
    print(f"训练样本数：{len(train_dataset)}")

    # 5. 创建 Trainer
    print("\n创建 Trainer（全量微调模式）")
    trainer = create_full_finetune_trainer(model, tokenizer, train_dataset, config)

    # 6. 开始训练
    print("\n开始全量微调...")
    print("=" * 60)
    trainer.train()
    print("=" * 60)

    # 7. 保存
    output_dir = config.training.output_dir
    print(f"\n保存完整模型到：{output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 计算保存大小
    import os
    total_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, f))
    )
    print(f"保存大小：{total_size / (1024*1024):.1f} MB")

    print("\n全量微调完成!")

    return model, tokenizer


def compare_lora_vs_full():
    """
    打印 LoRA 与全量微调的对比
    """
    print("""
╔════════════════════════════════════════════════════════╗
║           LoRA vs 全量微调 对比                         ║
╠════════════════════════════════════════════════════════╣
║  特性              │  LoRA      │  全量微调            ║
╠════════════════════════════════════════════════════════╣
║  可训练参数        │  0.1-1%    │  100%                ║
║  显存占用          │  低        │  高（4-8 倍）          ║
║  训练速度          │  快        │  慢                  ║
║  保存大小          │  10-100MB  │  几 GB               ║
║  适合场景          │  快速迭代  │  领域深度适配        ║
║  推荐模型大小      │  7B+       │  <1B                 ║
║  学习率            │  1e-4      │  1e-5                ║
╚════════════════════════════════════════════════════════╝

推荐：
- 入门学习：LoRA（快速验证，资源要求低）
- 小模型（<1B）：全量微调（参数量不大，效果可能更好）
- 大模型（>7B）：LoRA/QLoRA（唯一可行方案）
""")


if __name__ == "__main__":
    from config import Config

    # 显示对比
    compare_lora_vs_full()

    # 示例配置
    config = Config()
    config.training.data_path = "data.jsonl"
    config.training.output_dir = "./full_output"
    config.training.num_epochs = 3
    config.training.batch_size = 2  # 全量微调需要更小的 batch size

    print("\n准备开始全量微调...")
    print("警告：全量微调需要大量显存！")
    print("      如遇到 OOM，请减小 batch_size 或使用 LoRA")

    input("\n按 Enter 继续，或 Ctrl+C 退出...")

    train_full(config)
