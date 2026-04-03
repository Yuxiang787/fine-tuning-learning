# 01 - 极简入门

> 从最简单的单文件脚本开始，10 分钟内跑通第一个大模型微调

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行微调

```bash
python lora_finetune.py
```

### 3. 测试推理

```bash
python inference.py
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `lora_finetune.py` | 单文件 LoRA 微调脚本 |
| `inference.py` | 推理测试脚本 |
| `data.jsonl` | 示例训练数据（20 条指令） |
| `requirements.txt` | Python 依赖 |

## 原理解析

### LoRA 是什么？

LoRA（Low-Rank Adaptation）是一种**参数高效微调**技术：

```
原始模型参数（冻结）+ LoRA 适配器（可训练）= 完整模型

传统微调：████████████████████ 全部可训练（100% 参数）
LoRA 微调：████████████████████ 基础参数冻结
           └── ▒▒▒▒ 低秩适配器 仅训练 0.1-1% 参数
```

### 为什么用 LoRA？

| 对比项 | 传统微调 | LoRA |
|--------|----------|------|
| 显存占用 | 高 | 低 60-80% |
| 训练速度 | 慢 | 快 2-3 倍 |
| 存储成本 | 每个任务 7GB+ | 每个任务 10-100MB |
| 切换任务 | 重新加载模型 | 仅切换适配器 |

### 训练流程

```
1. 加载数据 → 2. 加载模型 → 3. 配置 LoRA → 4. 训练 → 5. 保存
     ↓              ↓             ↓            ↓         ↓
  data.jsonl   OPT-125M    r=8, alpha=16   3 epochs  adapter/
```

## 代码解析

### 核心代码（仅 50 行）

```python
# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# 2. 配置 LoRA
lora_config = LoraConfig(
    r=8,              # 秩：控制表达能力
    lora_alpha=16,    # 缩放系数
    target_modules=None,  # 自动选择
)

# 3. 应用 LoRA
model = get_peft_model(model, lora_config)

# 4. 训练
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()

# 5. 保存
model.save_pretrained("./lora_output")
```

## 自定义配置

### 更换模型

编辑 `lora_finetune.py` 中的 `MODEL_NAME`：

```python
# 推荐教学模型（按大小排序）
MODEL_NAME = "facebook/opt-125m"       # 125M，最快
MODEL_NAME = "Qwen/Qwen2.5-0.5B"       # 0.5B，中文友好
MODEL_NAME = "TinyLlama/TinyLlama-1.1B" # 1.1B，效果更好
```

### 调整 LoRA 参数

```python
LORA_R = 8              # 尝试 4, 8, 16, 32（越大越强）
LORA_ALPHA = 16         # 通常是 r 的 2 倍
LORA_DROPOUT = 0.1      # 防止过拟合
```

### 调整训练参数

```python
BATCH_SIZE = 4          # 显存允许情况下调大
NUM_EPOCHS = 3          # 通常 3-5 轮足够
LEARNING_RATE = 2e-4    # LoRA 常用 1e-4 到 5e-4
```

## 预期输出

```
============================================================
大模型 LoRA 微调教学示例
============================================================

[1] 使用设备：cuda

[2] 加载数据：data.jsonl
    样本数量：20
    示例：{'instruction': '将以下中文翻译成英文', 'input': '你好，世界！', ...}

[3] 加载模型：facebook/opt-125m
    模型参数量：125,000,000

[4] 配置 LoRA 参数
    可训练参数：983,040 / 125,971,968 (0.78%)
    trainable params: 983,040 || all params: 125,971,968

[5] 处理数据集
    分词后样本形状：torch.Size([256])

[6] 配置训练参数

[7] 创建 Trainer

[8] 开始训练
============================================================
{'loss': 2.456, 'learning_rate': 0.0002, 'epoch': 0.33}
{'loss': 1.876, 'learning_rate': 0.0002, 'epoch': 0.67}
...
============================================================

[9] 保存 LoRA 权重到：./lora_output

============================================================
训练完成！
============================================================
```

## 常见问题

### Q: CPU 能跑吗？
A: 可以，但会很慢（约 10-30 分钟一轮）。建议使用：
- Google Colab（免费 T4 GPU）
- Kaggle Notebook
- 本地 GPU（显存 4GB+）

### Q: 显存不足怎么办？
A: 降低 `BATCH_SIZE` 或 `MAX_LENGTH`，或使用梯度累积：
```python
BATCH_SIZE = 1  # 最小化单批大小
gradient_accumulation_steps=8  # 累积 8 步
```

### Q: 如何用自己的数据训练？
A: 按以下格式准备 `data.jsonl`：
```json
{"instruction": "任务描述", "input": "输入内容", "output": "期望输出"}
```

### Q: 训练后效果不好？
A: 尝试：
1. 增加训练数据（至少 100 条）
2. 增加训练轮数（5-10 轮）
3. 调整学习率（1e-4 到 5e-4）
4. 增大 LoRA 秩（16 或 32）

## 下一步

完成本教程后，你可以：

1. 📦 **模块化项目** - 学习更清晰的项目结构
2. 🔧 **全量微调** - 理解完整参数微调
3. 🏗️ **完整工程** - 配置、日志、评测、部署

## 参考资源

- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [PEFT 文档](https://huggingface.co/docs/peft)
- [Transformers 文档](https://huggingface.co/docs/transformers)
