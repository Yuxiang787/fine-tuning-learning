#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
极简 LoRA 微调示例 - 单文件实现大模型微调

运行此脚本前，确保已安装依赖：
    pip install transformers peft datasets accelerate torch

用法：
    python lora_finetune.py

说明：
    - 使用 HuggingFace transformers 和 peft 库
    - 采用 LoRA 参数高效微调方法
    - 示例数据为指令微调格式（instruction, input, output）
    - 默认使用 tiny 模型快速演示，可修改 MODEL_NAME 换其他模型
"""

import json
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

# ==================== 配置区域 ====================

# 模型选择 - M4 24GB 推荐
# 可选：'facebook/opt-125m'(测试), 'Qwen/Qwen2.5-0.5B'(推荐), 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'(进阶)
MODEL_NAME = "Qwen/Qwen2.5-0.5B"

# LoRA 参数
LORA_R = 16             # LoRA 秩，M4 24GB 可用 16
LORA_ALPHA = 32         # LoRA 缩放系数 (2*r)
LORA_DROPOUT = 0.1      # Dropout 比率
TARGET_MODULES = None   # None 表示自动选择

# 训练参数 - M4 24GB 推荐
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
MAX_LENGTH = 512        # M4 24GB 可用 512

# 路径（相对脚本目录，避免从仓库根目录运行时报找不到文件）
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.jsonl"
OUTPUT_DIR = BASE_DIR / "lora_output"

# ==================== 工具函数 ====================

def load_data(data_path):
    """加载 JSONL 格式的训练数据"""
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_prompt(example):
    """将数据格式化为模型输入格式"""
    # 构建指令模板：Instruction + Input -> Output
    if example["input"]:
        text = f"""### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Output:
{example["output"]}
"""
    else:
        text = f"""### Instruction:
{example["instruction"]}

### Output:
{example["output"]}
"""
    return {"text": text}


# ==================== 主函数 ====================

def main():
    print("=" * 60)
    print("大模型 LoRA 微调教学示例")
    print("=" * 60)

    # 1. 检查设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"\n[1] 使用设备：{device}")
    if device.type == "cpu":
        print("    提示：CPU 训练较慢")
    elif device.type == "mps":
        print("    Apple Silicon MPS 加速已启用")

    # 2. 加载数据
    print(f"\n[2] 加载数据：{DATA_PATH}")
    raw_data = load_data(DATA_PATH)
    print(f"    样本数量：{len(raw_data)}")
    print(f"    示例：{raw_data[0]}")

    # 3. 加载模型和分词器
    print(f"\n[3] 加载模型：{MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 设置 padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device.type in ["cuda", "mps"] else torch.float32,
        device_map="auto" if device.type in ["cuda", "mps"] else None,
    )
    if device.type != "cuda":
        model = model.to(device)
    print(f"    模型参数量：{model.num_parameters():,}")

    # 4. 配置 LoRA
    print("\n[4] 配置 LoRA 参数")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    可训练参数：{trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    model.print_trainable_parameters()

    # 5. 处理数据
    print("\n[5] 处理数据集")
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(tokenize, batched=True)
    print(f"    分词后样本形状：{len(tokenized_dataset[0]['input_ids'])}")

    # 6. 训练配置
    print("\n[6] 配置训练参数")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",  # 禁用 wandb，如需启用可改为 "wandb"
        fp16=device.type in ["cuda", "mps"],
        gradient_accumulation_steps=2,
        warmup_steps=10,
    )

    # 数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 因果语言建模
    )

    # 7. 创建 Trainer
    print("\n[7] 创建 Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 8. 开始训练
    print("\n[8] 开始训练")
    print("=" * 60)
    trainer.train()
    print("=" * 60)

    # 9. 保存模型
    print(f"\n[9] 保存 LoRA 权重到：{OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"\n后续步骤：")
    print(f"  1. 查看训练输出：ls -la {OUTPUT_DIR}")
    print(f"  2. 使用适配器推理：见 inference.py 示例")
    print(f"  3. 合并权重导出：使用 merge_adapter.py")

    return model, tokenizer


if __name__ == "__main__":
    main()
