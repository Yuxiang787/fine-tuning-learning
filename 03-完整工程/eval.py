#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测脚本

用法:
    python eval.py --adapter ./output/lora_opt125m
    python eval.py --model ./output/full_opt125m
    python eval.py --data eval_data.jsonl --num_samples 20
"""

import sys
import argparse
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.eval.evaluator import Evaluator, load_eval_data


def main():
    parser = argparse.ArgumentParser(description="评测微调模型")

    parser.add_argument(
        "--base_model",
        type=str,
        default="facebook/opt-125m",
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
        help="微调后的模型路径（全量微调）"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data.jsonl",
        help="评测数据路径"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="评测样本数量"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="结果保存路径"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="生成温度"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="最大生成 token 数"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("模型评测")
    print("=" * 60)

    # 加载评测数据
    print(f"\n加载评测数据：{args.data}")
    eval_data = load_eval_data(args.data, args.num_samples)
    print(f"样本数量：{len(eval_data)}")

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

    # 评测
    print(f"\n评测中...")
    results = evaluator.evaluate_batch(eval_data)

    # 打印结果
    evaluator.print_results(results)

    # 保存结果
    if args.output:
        evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()
