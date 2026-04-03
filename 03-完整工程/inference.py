#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理脚本 - 与微调模型交互

用法:
    python inference.py --adapter ./output/lora_qwen0.5b
    python inference.py --model ./output/full_qwen0.5b
"""

import sys
import argparse
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.eval.evaluator import Evaluator


def interactive_mode(evaluator: Evaluator):
    """交互模式"""
    print("\n" + "=" * 60)
    print("交互模式 - 输入 q 退出")
    print("=" * 60)

    while True:
        try:
            instruction = input("\n指令：").strip()
            if instruction.lower() in ['q', 'quit', 'exit']:
                print("再见!")
                break

            input_text = input("输入（可选，直接回车跳过）：").strip()

            response = evaluator.generate(
                instruction,
                input_text if input_text else "",
                temperature=0.7,
                max_new_tokens=256
            )

            print(f"回复：{response}")

        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except Exception as e:
            print(f"错误：{e}")


def demo_mode(evaluator: Evaluator):
    """演示模式 - 运行预设测试"""
    test_cases = [
        ("将以下中文翻译成英文", "我喜欢编程"),
        ("判断以下句子的情感倾向", "这个产品很好用"),
        ("回答以下数学问题", "25 + 17 = ?"),
        ("解释以下概念", "什么是机器学习？"),
        ("续写以下故事", "从前有座山，"),
    ]

    print("\n" + "=" * 60)
    print("演示测试")
    print("=" * 60)

    for instruction, input_text in test_cases:
        print(f"\n指令：{instruction}")
        print(f"输入：{input_text}")
        response = evaluator.generate(
            instruction,
            input_text,
            temperature=0.7,
            max_new_tokens=100
        )
        print(f"回复：{response}")
        print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="推理脚本")

    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="基础模型名称"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="LoRA 适配器路径"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="微调后的模型路径"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="运行演示测试"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="进入交互模式"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("模型推理")
    print("=" * 60)

    # 加载模型
    if args.adapter:
        print(f"\n加载 LoRA 模型:")
        print(f"  基础模型：{args.base_model}")
        print(f"  适配器：{args.adapter}")
        evaluator = Evaluator.from_lora(args.base_model, args.adapter)
    elif args.model:
        print(f"\n加载微调模型：{args.model}")
        evaluator = Evaluator.from_finetuned(args.model)
    else:
        print("错误：请指定 --adapter 或 --model")
        sys.exit(1)

    print("模型加载完成!")

    # 运行模式
    if args.demo:
        demo_mode(evaluator)

    if args.interactive or (not args.demo):
        interactive_mode(evaluator)


if __name__ == "__main__":
    main()
