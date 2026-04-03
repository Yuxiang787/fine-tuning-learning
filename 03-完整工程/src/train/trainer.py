#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练器模块
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from ..models import load_model, load_tokenizer, create_lora_config, apply_lora
from ..data import FineTuningDataset


logger = logging.getLogger(__name__)


def create_training_args(
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    warmup_steps: int = 10,
    logging_steps: int = 10,
    save_strategy: str = "epoch",
    save_steps: int = 500,
    gradient_accumulation_steps: int = 2,
    fp16: bool = True,
    bf16: bool = False,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    dataloader_workers: int = 0,
    seed: int = 42,
    report_to: str = "none",
    run_name: str = "finetune",
) -> TrainingArguments:
    """
    创建训练参数

    Args:
        output_dir: 输出目录
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        warmup_steps: 预热步数
        logging_steps: 日志步数
        save_strategy: 保存策略
        save_steps: 保存步数
        gradient_accumulation_steps: 梯度累积步数
        fp16: 是否使用 FP16
        bf16: 是否使用 BF16
        weight_decay: 权重衰减
        max_grad_norm: 最大梯度范数
        dataloader_workers: 数据加载进程数
        seed: 随机种子
        report_to: 报告工具
        run_name: 运行名称

    Returns:
        TrainingArguments
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=fp16,
        bf16=bf16,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        dataloader_num_workers=dataloader_workers,
        seed=seed,
        report_to=report_to,
        run_name=run_name,
        logging_dir=f"{output_dir}/logs",
    )


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset=None,
    training_args: Optional[TrainingArguments] = None,
    data_collator=None,
) -> Trainer:
    """
    创建 Trainer

    Args:
        model: 模型
        tokenizer: 分词器
        train_dataset: 训练数据集
        eval_dataset: 验证数据集
        training_args: 训练参数
        data_collator: 数据收集器

    Returns:
        Trainer
    """
    if data_collator is None:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )


def train(args: Dict[str, Any]) -> tuple:
    """
    执行训练

    Args:
        args: 参数字典

    Returns:
        (模型，分词器)
    """
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 创建设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备：{device}")

    # 加载分词器
    logger.info(f"加载分词器：{args.get('tokenizer_name') or args['model_name']}")
    tokenizer = load_tokenizer(
        args.get('tokenizer_name') or args['model_name'],
        trust_remote_code=args.get('trust_remote_code', True)
    )

    # 加载模型
    logger.info(f"加载模型：{args['model_name']}")
    model, device = load_model(
        args['model_name'],
        device=device,
        trust_remote_code=args.get('trust_remote_code', True),
    )

    # 应用 LoRA（如果启用）
    if args.get('use_lora', True) and not args.get('full_finetune', False):
        logger.info("应用 LoRA 适配器")
        lora_config = create_lora_config(
            r=args.get('lora_r', 8),
            alpha=args.get('lora_alpha', 16),
            dropout=args.get('lora_dropout', 0.1),
            target_modules=args.get('target_modules'),
        )
        model, trainable, total = apply_lora(model, lora_config)
        logger.info(f"可训练参数：{trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    else:
        logger.info("使用全量微调模式")

    # 加载数据
    logger.info(f"加载训练数据：{args['data']}")
    format_template = """### Instruction:
{instruction}

### Input:
{input}

### Output:
{output}
"""
    format_template_no_input = """### Instruction:
{instruction}

### Output:
{output}
"""

    train_dataset_obj = FineTuningDataset(
        data_path=args['data'],
        format_template=format_template,
        format_template_no_input=format_template_no_input
    )
    train_dataset_obj.load().format().tokenize(tokenizer, args.get('max_length', 256))
    train_dataset = train_dataset_obj.to_huggingface()

    logger.info(f"训练样本数：{len(train_dataset)}")

    # 加载验证数据（如果有）
    eval_dataset = None
    if args.get('validation_data'):
        logger.info(f"加载验证数据：{args['validation_data']}")
        eval_dataset_obj = FineTuningDataset(
            data_path=args['validation_data'],
            format_template=format_template,
            format_template_no_input=format_template_no_input
        )
        eval_dataset_obj.load().format().tokenize(tokenizer, args.get('max_length', 256))
        eval_dataset = eval_dataset_obj.to_huggingface()

    # 创建训练参数
    training_args = create_training_args(
        output_dir=args.get('output_dir', './output'),
        num_epochs=args.get('num_epochs', 3),
        batch_size=args.get('batch_size', 4),
        learning_rate=args.get('learning_rate', 2e-4),
        warmup_steps=args.get('warmup_steps', 10),
        logging_steps=args.get('logging_steps', 10),
        save_strategy=args.get('save_strategy', 'epoch'),
        gradient_accumulation_steps=args.get('gradient_accumulation_steps', 2),
        fp16=args.get('fp16', True),
        weight_decay=args.get('weight_decay', 0.01),
        seed=args.get('seed', 42),
        report_to=args.get('report_to', 'none'),
        run_name=args.get('run_name', 'finetune'),
    )

    # 创建 Trainer
    logger.info("创建 Trainer")
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=training_args,
    )

    # 开始训练
    logger.info("开始训练...")
    print("\n" + "=" * 60)
    trainer.train(resume_from_checkpoint=args.get('resume'))
    print("=" * 60)

    # 保存
    output_dir = args.get('output_dir', './output')
    logger.info(f"保存模型到：{output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("训练完成!")

    return model, tokenizer
