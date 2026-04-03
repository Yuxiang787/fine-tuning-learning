#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测模块 - 评估微调模型的效果
"""

import torch
import json
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re


class Evaluator:
    """模型评测器"""

    def __init__(self, model, tokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @classmethod
    def load_lora_model(
        cls,
        base_model_name: str,
        adapter_path: str
    ) -> "Evaluator":
        """加载 LoRA 模型"""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        # transformers >= 4.48.0 supports device_map="auto" on MPS
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device.type in ["cuda", "mps"] else torch.float32,
            device_map="auto" if device.type in ["cuda", "mps"] else None,
        )

        if device.type not in ["cuda", "mps"]:
            base_model = base_model.to(device)

        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()

        return cls(model, tokenizer, device)

    def generate(
        self,
        instruction: str,
        input_text: str = "",
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """生成回复"""
        # 构建提示
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
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # 生成 - 使用 torch.amp.autocast for MPS (torch >= 2.10.0)
        if self.device.type == "mps":
            with torch.autocast(device_type="mps", dtype=torch.float16):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        else:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        # 解码
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_text[len(prompt):].strip()

        return response

    def evaluate_batch(
        self,
        examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """批量评测"""
        results = []

        for i, example in enumerate(examples):
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            expected = example.get("output", "")

            generated = self.generate(instruction, input_text)

            results.append({
                "id": i,
                "instruction": instruction,
                "input": input_text,
                "expected": expected,
                "generated": generated,
                "match": self._check_match(expected, generated),
            })

        return results

    def _check_match(self, expected: str, generated: str) -> bool:
        """检查生成结果是否匹配（宽松匹配）"""
        # 去除标点符号和空白
        expected_clean = re.sub(r'[^\w\u4e00-\u9fff]', '', expected.lower())
        generated_clean = re.sub(r'[^\w\u4e00-\u9fff]', '', generated.lower())

        # 完全匹配
        if expected_clean == generated_clean:
            return True

        # 包含匹配（生成的包含期望的关键词）
        if len(expected_clean) > 2 and expected_clean in generated_clean:
            return True

        return False

    def print_results(self, results: List[Dict[str, Any]]):
        """打印评测结果"""
        print("\n" + "=" * 60)
        print("评测结果")
        print("=" * 60)

        match_count = sum(1 for r in results if r["match"])
        total = len(results)
        accuracy = match_count / total if total > 0 else 0

        print(f"\n匹配率：{match_count}/{total} ({accuracy:.1%})")
        print("-" * 60)

        for r in results:
            status = "✓" if r["match"] else "✗"
            print(f"\n[{status}] 指令：{r['instruction']}")
            if r['input']:
                print(f"    输入：{r['input']}")
            print(f"    期望：{r['expected']}")
            print(f"    生成：{r['generated']}")

        print("\n" + "=" * 60)


def load_test_data(path: str) -> List[Dict[str, Any]]:
    """加载测试数据"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="评测微调模型")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--adapter", type=str, default="./lora_output")
    parser.add_argument("--data", type=str, default="data.jsonl")
    parser.add_argument("--num-samples", type=int, default=10)
    args = parser.parse_args()

    print("=" * 60)
    print("模型评测")
    print("=" * 60)
    print(f"基础模型：{args.base_model}")
    print(f"适配器：{args.adapter}")
    print(f"测试数据：{args.data}")

    # 加载模型
    print("\n加载模型...")
    evaluator = Evaluator.load_lora_model(args.base_model, args.adapter)

    # 加载测试数据
    print("加载测试数据...")
    test_data = load_test_data(args.data)[:args.num_samples]

    # 评测
    print(f"评测 {len(test_data)} 个样本...")
    results = evaluator.evaluate_batch(test_data)

    # 打印结果
    evaluator.print_results(results)


if __name__ == "__main__":
    main()
