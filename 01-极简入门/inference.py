#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA 微调后推理示例

用法：
    python inference.py
"""

from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==================== 配置 ====================

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
BASE_DIR = Path(__file__).resolve().parent
LORA_PATH = BASE_DIR / "lora_output"

# ==================== 推理函数 ====================

def load_model():
    """加载基础模型和 LoRA 适配器"""
    adapter_config = LORA_PATH / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(
            f"未找到 LoRA 适配器配置文件: {adapter_config}\n"
            f"请先运行训练脚本生成适配器，或检查 LORA_PATH 是否正确。"
        )

    print("加载基础模型...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else torch.float32,
        device_map="auto" if (torch.cuda.is_available() or torch.backends.mps.is_available()) else None,
    )

    print("加载 LoRA 适配器...")
    model = PeftModel.from_pretrained(model, LORA_PATH)

    # 如需合并权重导出，可取消注释
    # model = model.merge_and_unload()

    return model, tokenizer


def generate_response(model, tokenizer, instruction, input_text=""):
    """生成回复"""
    # 构建输入提示
    if input_text:
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Output:
"""
    else:
        prompt = f"""### Instruction:
{instruction}

### Output:
"""

    # 分词
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 生成 - 使用 torch.amp.autocast for MPS (torch >= 2.10.0)
    do_sample = True
    if model.device.type == "mps":
        with torch.autocast(device_type="mps", dtype=torch.float16):
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=do_sample,
                temperature=0.7,
                top_p=0.9,
            )
    else:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=do_sample,
                temperature=0.7,
                top_p=0.9,
            )

    # 解码结果（只输出新生成部分）
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_text[len(prompt):].strip()

    return response


def main():
    print("=" * 60)
    print("LoRA 微调模型推理演示")
    print("=" * 60)

    # 加载模型
    model, tokenizer = load_model()
    model.eval()

    print("\n模型加载完成，开始测试...\n")

    # 测试用例
    test_cases = [
        ("将以下中文翻译成英文", "我喜欢编程"),
        ("判断以下句子的情感倾向", "这个产品很好用"),
        ("回答以下数学问题", "25 + 17 = ?"),
        ("解释以下概念", "什么是机器学习？"),
    ]

    for instruction, input_text in test_cases:
        print(f"指令：{instruction}")
        print(f"输入：{input_text}")
        response = generate_response(model, tokenizer, instruction, input_text)
        print(f"输出：{response}")
        print("-" * 40)

    # 交互式输入
    print("\n进入交互模式（输入 q 退出）")
    while True:
        instruction = input("\n请输入指令：").strip()
        if instruction.lower() == "q":
            break
        input_text = input("请输入内容（可选）：").strip()

        response = generate_response(model, tokenizer, instruction, input_text)
        print(f"回复：{response}")


if __name__ == "__main__":
    main()
