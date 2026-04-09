#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练器模块
"""

import logging
import os
from datetime import datetime
from math import ceil
from typing import Optional, Dict, Any

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
)

from ..models import load_model, load_tokenizer, create_lora_config, apply_lora
from ..data import FineTuningDataset


logger = logging.getLogger(__name__)


def build_tensorboard_log_dir(output_dir: str, run_name: str) -> str:
    """为 TensorBoard 构建稳定且易区分的日志目录"""
    safe_run_name = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in (run_name or "").strip()
    ).strip("_")

    if not safe_run_name:
        safe_run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    return f"{output_dir}/logs/{safe_run_name}"


def configure_library_logging():
    """降低第三方库的低信号日志噪音"""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def format_dtype(dtype: Optional[torch.dtype]) -> str:
    """Format torch dtype for logs."""
    if dtype is None:
        return "auto"
    return str(dtype).replace("torch.", "")


def log_runtime_configuration(
    args: Dict[str, Any],
    device: torch.device,
    model_dtype: Optional[torch.dtype],
    train_dataset_size: int,
):
    """Log the effective runtime configuration in English."""
    use_lora = args.get("use_lora", True) and not args.get("full_finetune", False)
    mode_name = "LoRA" if use_lora else "Full fine-tuning"
    per_device_batch = args.get("batch_size", 8)
    accumulation = args.get("gradient_accumulation_steps", 2)
    effective_batch = per_device_batch * accumulation
    num_epochs = args.get("num_epochs", 3)
    steps_per_epoch = ceil(train_dataset_size / effective_batch) if train_dataset_size > 0 else 0
    total_steps = steps_per_epoch * num_epochs

    logger.info("=" * 60)
    logger.info("Runtime configuration")
    logger.info("=" * 60)
    logger.info("Mode: %s", mode_name)
    logger.info("Device: %s", device)
    logger.info("Model: %s", args["model_name"])
    logger.info("Data path: %s", args["data"])
    logger.info("Output directory: %s", args.get("output_dir", "./output"))
    logger.info("Model dtype: %s", format_dtype(model_dtype))
    logger.info("Trainer precision: %s", "fp16" if args.get("fp16", True) and device.type == "cuda" else "fp32")
    logger.info("Per-device batch size: %s", per_device_batch)
    logger.info("Gradient accumulation steps: %s", accumulation)
    logger.info("Effective batch size: %s", effective_batch)
    logger.info("Max sequence length: %s", args.get("max_length", 512))
    logger.info("Learning rate: %s", args.get("learning_rate", 2e-4))
    logger.info("Epochs: %s", num_epochs)
    logger.info("Training samples: %s", train_dataset_size)
    logger.info("Steps per epoch: %s", steps_per_epoch)
    logger.info("Total training steps: %s", total_steps)
    logger.info("Logging backend: %s", args.get("report_to", "tensorboard"))
    logger.info("Run name: %s", args.get("run_name", "lora-qwen0.5b-m4"))
    if use_lora:
        logger.info(
            "LoRA config: r=%s, alpha=%s, dropout=%s",
            args.get("lora_r", 16),
            args.get("lora_alpha", 32),
            args.get("lora_dropout", 0.1),
        )
    logger.info("=" * 60)


class TrainingProgressCallback(TrainerCallback):
    """Emit concise English logs during training."""

    def on_train_begin(self, args, state, control, **kwargs):
        logger.info(
            "Training started: epochs=%s, max_steps=%s, logging_steps=%s",
            state.num_train_epochs,
            state.max_steps,
            args.logging_steps,
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        metrics = []
        for key in ("loss", "grad_norm", "learning_rate", "epoch"):
            if key in logs:
                metrics.append(f"{key}={logs[key]}")

        if metrics:
            logger.info(
                "Training progress: step=%s/%s | %s",
                state.global_step,
                state.max_steps,
                " | ".join(metrics),
            )

    def on_epoch_end(self, args, state, control, **kwargs):
        latest_loss = None
        latest_lr = None

        for record in reversed(state.log_history or []):
            if latest_loss is None and "loss" in record:
                latest_loss = record["loss"]
            if latest_lr is None and "learning_rate" in record:
                latest_lr = record["learning_rate"]
            if latest_loss is not None and latest_lr is not None:
                break

        summary = [f"epoch={state.epoch}"]
        if latest_loss is not None:
            summary.append(f"loss={latest_loss}")
        if latest_lr is not None:
            summary.append(f"learning_rate={latest_lr}")

        logger.info("Epoch complete: %s", " | ".join(summary))

    def on_train_end(self, args, state, control, **kwargs):
        final_train_loss = None
        final_epoch = state.epoch

        for record in reversed(state.log_history or []):
            if "train_loss" in record:
                final_train_loss = record["train_loss"]
                final_epoch = record.get("epoch", final_epoch)
                break

        logger.info(
            "Training finished: global_step=%s, epoch=%s, train_loss=%s",
            state.global_step,
            final_epoch,
            final_train_loss,
        )


def create_training_args(
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 8,
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
    report_to: str = "tensorboard",
    run_name: str = "lora-qwen0.5b-m4",
    device_type: str = "cpu",
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
    use_fp16 = fp16 and device_type == "cuda"
    use_bf16 = bf16 and device_type == "cuda"

    if report_to == "tensorboard":
        os.environ["TENSORBOARD_LOGGING_DIR"] = build_tensorboard_log_dir(output_dir, run_name)

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_strategy="epoch",
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=use_fp16,
        bf16=use_bf16,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        dataloader_num_workers=dataloader_workers,
        dataloader_pin_memory=device_type == "cuda",
        seed=seed,
        report_to=report_to,
        run_name=run_name,
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
        processing_class=tokenizer,
        callbacks=[TrainingProgressCallback],
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
    configure_library_logging()

    # 创建设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # 加载分词器
    logger.info("Loading tokenizer: %s", args.get('tokenizer_name') or args['model_name'])
    tokenizer = load_tokenizer(
        args.get('tokenizer_name') or args['model_name'],
        trust_remote_code=args.get('trust_remote_code', True)
    )

    # 加载模型
    logger.info("Loading model: %s", args['model_name'])
    is_full_finetune = args.get('full_finetune', False) or not args.get('use_lora', True)
    model_dtype = None
    if device.type == "mps":
        model_dtype = torch.float32 if is_full_finetune else torch.float16

    model, device = load_model(
        args['model_name'],
        device=device,
        trust_remote_code=args.get('trust_remote_code', True),
        dtype=model_dtype,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False

    # 应用 LoRA（如果启用）
    if args.get('use_lora', True) and not args.get('full_finetune', False):
        logger.info("Applying LoRA adapters")
        lora_config = create_lora_config(
            r=args.get('lora_r', 16),
            alpha=args.get('lora_alpha', 32),
            dropout=args.get('lora_dropout', 0.1),
            target_modules=args.get('target_modules'),
        )
        model, trainable, total = apply_lora(model, lora_config)
        logger.info("Trainable parameters: %s / %s (%.2f%%)", f"{trainable:,}", f"{total:,}", 100 * trainable / total)
    else:
        logger.info("Using full fine-tuning mode")

    # 加载数据
    logger.info("Loading training data: %s", args['data'])
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
    train_dataset_obj.load().format().tokenize(tokenizer, args.get('max_length', 512))
    train_dataset = train_dataset_obj.to_huggingface()

    logger.info("Training samples loaded: %s", len(train_dataset))

    # 加载验证数据（如果有）
    eval_dataset = None
    if args.get('validation_data'):
        logger.info("Loading validation data: %s", args['validation_data'])
        eval_dataset_obj = FineTuningDataset(
            data_path=args['validation_data'],
            format_template=format_template,
            format_template_no_input=format_template_no_input
        )
        eval_dataset_obj.load().format().tokenize(tokenizer, args.get('max_length', 512))
        eval_dataset = eval_dataset_obj.to_huggingface()

    # 创建训练参数
    training_args = create_training_args(
        output_dir=args.get('output_dir', './output'),
        num_epochs=args.get('num_epochs', 3),
        batch_size=args.get('batch_size', 8),
        learning_rate=args.get('learning_rate', 2e-4),
        warmup_steps=args.get('warmup_steps', 10),
        logging_steps=args.get('logging_steps', 10),
        save_strategy=args.get('save_strategy', 'epoch'),
        gradient_accumulation_steps=args.get('gradient_accumulation_steps', 2),
        fp16=args.get('fp16', True),
        bf16=args.get('bf16', False),
        weight_decay=args.get('weight_decay', 0.01),
        seed=args.get('seed', 42),
        report_to=args.get('report_to', 'tensorboard'),
        run_name=args.get('run_name', 'lora-qwen0.5b-m4'),
        device_type=device.type,
    )

    # 创建 Trainer
    log_runtime_configuration(
        args=args,
        device=device,
        model_dtype=model_dtype,
        train_dataset_size=len(train_dataset),
    )
    logger.info("Creating Trainer")
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=training_args,
    )

    # 开始训练
    logger.info("Starting training")
    print("\n" + "=" * 60)
    trainer.train(resume_from_checkpoint=args.get('resume'))
    print("=" * 60)

    # 保存
    output_dir = args.get('output_dir', './output')
    logger.info("Saving model to: %s", output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Training complete")

    return model, tokenizer
