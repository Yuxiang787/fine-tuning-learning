#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA 微调主模块
"""

import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

from config import Config
from model import (
    load_tokenizer,
    load_base_model,
    create_lora_config,
    apply_lora,
    print_model_info,
    get_device,
)
from data import create_training_dataset


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    config: Config
) -> Trainer:
    """
    创建 Trainer

    Args:
        model: 模型
        tokenizer: 分词器
        train_dataset: 训练数据集
        config: 配置

    Returns:
        Trainer 实例
    """
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        logging_steps=config.training.logging_steps,
        save_strategy=config.training.save_strategy,
        save_total_limit=config.training.save_total_limit,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        fp16=config.training.fp16 and get_device().type == "cuda",
        dataloader_num_workers=config.training.dataloader_num_workers,
        seed=config.training.seed,
        report_to=config.training.report_to,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 因果语言建模
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )


def train_lora(config: Config) -> tuple:
    """
    执行 LoRA 微调训练

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
    print_model_info(model, "基础模型")

    # 3. 应用 LoRA
    print("\n应用 LoRA 适配器")
    lora_config = create_lora_config(config.lora)
    model, trainable_params, total_params = apply_lora(model, lora_config)
    print(f"可训练参数：{trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

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
    print("\n创建 Trainer")
    trainer = create_trainer(model, tokenizer, train_dataset, config)

    # 6. 开始训练
    print("\n开始训练...")
    print("=" * 60)
    trainer.train()
    print("=" * 60)

    # 7. 保存
    output_dir = config.training.output_dir
    print(f"\n保存模型到：{output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n训练完成!")

    return model, tokenizer


if __name__ == "__main__":
    # 示例：使用默认配置训练
    from config import Config

    config = Config()
    config.training.data_path = "data.jsonl"
    config.training.output_dir = "./lora_output"
    config.training.num_epochs = 3
    config.training.batch_size = 4

    train_lora(config)
