#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测模块 - 评估微调模型效果
"""

import json
import re
import torch
from typing import List, Dict, Any
from pathlib import Path

from ..models import load_peft_model, load_model, load_tokenizer


class Evaluator:
    """模型评测器"""

    def __init__(self, model, tokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @classmethod
    def from_lora(
        cls,
        base_model_name: str,
        adapter_path: str
    ) -> "Evaluator":
        """从 LoRA 适配器加载"""
        model, tokenizer, device = load_peft_model(base_model_name, adapter_path)
        return cls(model, tokenizer, device)

    @classmethod
    def from_finetuned(
        cls,
        model_path: str
    ) -> "Evaluator":
        """从微调后的模型加载"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = load_tokenizer(model_path)
        model, _ = load_model(model_path, device)
        model.eval()
        return cls(model, tokenizer, device)

    def generate(
        self,
        instruction: str,
        input_text: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        生成回复

        Args:
            instruction: 指令
            input_text: 输入文本
            max_new_tokens: 最大生成 token 数
            temperature: 温度
            top_p: Top-p 采样
            do_sample: 是否采样

        Returns:
            生成的回复
        """
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

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_text[len(prompt):].strip()

        return response

    def evaluate_batch(
        self,
        examples: List[Dict[str, Any]],
        match_keywords: bool = True
    ) -> List[Dict[str, Any]]:
        """
        批量评测

        Args:
            examples: 评测样本列表
            match_keywords: 是否使用关键词匹配

        Returns:
            评测结果列表
        """
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
                "exact_match": self._exact_match(expected, generated),
                "keyword_match": self._keyword_match(expected, generated) if match_keywords else None,
            })

        return results

    def _exact_match(self, expected: str, generated: str) -> bool:
        """精确匹配检查"""
        expected_clean = re.sub(r'[^\w\u4e00-\u9fff]', '', expected.lower())
        generated_clean = re.sub(r'[^\w\u4e00-\u9fff]', '', generated.lower())
        return expected_clean == generated_clean

    def _keyword_match(self, expected: str, generated: str) -> bool:
        """关键词匹配检查"""
        expected_clean = re.sub(r'[^\w\u4e00-\u9fff]', '', expected.lower())
        generated_clean = re.sub(r'[^\w\u4e00-\u9fff]', '', generated.lower())

        if len(expected_clean) > 2:
            return expected_clean in generated_clean
        return False

    def print_results(self, results: List[Dict[str, Any]]):
        """打印评测结果"""
        print("\n" + "=" * 60)
        print("评测结果")
        print("=" * 60)

        exact_matches = sum(1 for r in results if r["exact_match"])
        keyword_matches = sum(1 for r in results if r.get("keyword_match"))
        total = len(results)

        print(f"\n精确匹配：{exact_matches}/{total} ({100*exact_matches/total:.1f}%)")
        if keyword_matches:
            print(f"关键词匹配：{keyword_matches}/{total} ({100*keyword_matches/total:.1f}%)")

        print("-" * 60)

        for r in results:
            status = "✓" if r["exact_match"] else ("~" if r.get("keyword_match") else "✗")
            print(f"\n[{status}] 指令：{r['instruction']}")
            if r['input']:
                print(f"    输入：{r['input']}")
            print(f"    期望：{r['expected']}")
            print(f"    生成：{r['generated']}")

        print("\n" + "=" * 60)

    def save_results(self, results: List[Dict[str, Any]], path: str):
        """保存评测结果"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到：{path}")


def load_eval_data(path: str, limit: int = None) -> List[Dict[str, Any]]:
    """
    加载评测数据

    Args:
        path: 数据文件路径
        limit: 限制样本数量

    Returns:
        样本列表
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    if limit:
        data = data[:limit]

    return data
