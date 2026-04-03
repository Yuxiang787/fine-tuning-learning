# 02 - 模块化项目

> 将代码拆分为可复用的模块，支持 LoRA 和全量微调两种模式

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行训练

```bash
# LoRA 微调（默认）
python train.py

# 全量微调
python train.py --full

# 使用自定义参数
python train.py --model Qwen/Qwen2.5-0.5B --epochs 5 --batch-size 8
```

### 3. 评测模型

```bash
python evaluate.py --adapter ./output
```

## 项目结构

```
02-模块化项目/
├── config.py           # 配置管理（数据类）
├── data.py             # 数据加载与处理
├── model.py            # 模型加载与 LoRA 配置
├── lora.py             # LoRA 微调实现
├── full_finetune.py    # 全量微调实现
├── train.py            # 训练主入口
├── evaluate.py         # 评测脚本
├── data.jsonl          # 示例数据
└── requirements.txt    # 依赖
```

## 模块说明

### config.py - 配置管理

使用 Python 数据类管理所有超参数：

```python
from config import Config

config = Config()
config.model.model_name = "Qwen/Qwen2.5-0.5B"
config.training.num_epochs = 5
config.lora.r = 16

# 保存/加载配置
config.save("my_config.json")
config = Config.load("my_config.json")
```

**配置类别：**
- `ModelConfig` - 模型选择、信任远程代码等
- `LoraConfig` - LoRA 秩、alpha、dropout 等
- `TrainingConfig` - 学习率、batch size、轮数等
- `DataConfig` - 数据列名、提示模板等

### data.py - 数据处理

**核心函数：**
- `load_jsonl()` - 加载 JSONL 数据
- `create_training_dataset()` - 创建训练数据集
- `preprocess_for_sft()` - SFT 预处理（带 label 掩码）

**示例：**
```python
from data import create_training_dataset
from config import default_config

dataset = create_training_dataset(
    data_path="data.jsonl",
    config=default_config.data,
    tokenizer=tokenizer,
    max_length=256
)
```

### model.py - 模型加载

**核心函数：**
- `load_tokenizer()` - 加载分词器
- `load_base_model()` - 加载基础模型
- `apply_lora()` - 应用 LoRA 适配器
- `merge_lora_weights()` - 合并 LoRA 权重

**示例：**
```python
from model import load_tokenizer, load_base_model, apply_lora

tokenizer = load_tokenizer("Qwen/Qwen2.5-0.5B")
model = load_base_model("Qwen/Qwen2.5-0.5B")
model, trainable, total = apply_lora(model, lora_config)
```

### lora.py - LoRA 微调

封装完整的 LoRA 微调流程：

```python
from config import Config
from lora import train_lora

config = Config()
config.training.data_path = "data.jsonl"
config.training.output_dir = "./output"

model, tokenizer = train_lora(config)
```

### full_finetune.py - 全量微调

全量微调所有参数：

```python
from full_finetune import train_full

config = Config()
config.training.batch_size = 2  # 全量微调需要更小的 batch

model, tokenizer = train_full(config)
```

### train.py - 训练入口

统一的命令行入口：

```bash
# 帮助
python train.py -h

# LoRA 微调
python train.py --model Qwen/Qwen2.5-0.5B --epochs 5

# 全量微调
python train.py --full --batch-size 2

# 使用配置文件
python train.py --config config.json
```

### evaluate.py - 评测

评估模型生成质量：

```bash
python evaluate.py --adapter ./output --num-samples 20
```

**评测指标：**
- 精确匹配率
- 关键词包含匹配
- 逐样本详细对比

## 命令行参数

### train.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--full` | 全量微调 | 否 (LoRA) |
| `--config` | 配置文件路径 | - |
| `--data` | 训练数据 | data.jsonl |
| `--output` | 输出目录 | ./output |
| `--model` | 模型名称 | Qwen/Qwen2.5-0.5B |
| `--epochs` | 训练轮数 | 3 |
| `--batch-size` | 批次大小 | 8 |
| `--lr` | 学习率 | 2e-4 |
| `--lora-r` | LoRA 秩 | 16 |

### evaluate.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--base-model` | 基础模型 | Qwen/Qwen2.5-0.5B |
| `--adapter` | 适配器路径 | ./lora_output |
| `--data` | 测试数据 | data.jsonl |
| `--num-samples` | 评测样本数 | 10 |

## 配置示例

### 保存配置到文件

```python
from config import Config

config = Config()
config.model.model_name = "Qwen/Qwen2.5-0.5B"
config.lora.r = 16
config.lora.alpha = 32
config.training.num_epochs = 5
config.training.batch_size = 8

config.save("my_config.json")
```

### 从文件加载配置

```python
from config import Config

config = Config.load("my_config.json")

# 直接使用
from lora import train_lora
train_lora(config)
```

## 对比：LoRA vs 全量微调

运行内置对比：

```bash
python full_finetune.py
```

输出：
```
╔════════════════════════════════════════════════════════╗
║           LoRA vs 全量微调 对比                         ║
╠════════════════════════════════════════════════════════╣
║  特性              │  LoRA      │  全量微调            ║
╠════════════════════════════════════════════════════════╣
║  可训练参数        │  0.1-1%    │  100%                ║
║  显存占用          │  低        │  高（4-8 倍）          ║
║  训练速度          │  快        │  慢                  ║
║  保存大小          │  10-100MB  │  几 GB               ║
║  适合场景          │  快速迭代  │  领域深度适配        ║
╚════════════════════════════════════════════════════════╝
```

## 进阶用法

### 自定义提示模板

```python
from config import Config

config = Config()
config.data.prompt_template = """<|user|>
{instruction}

<|assistant|>
{output}
"""

# 或者无 input 的模板
config.data.prompt_template_no_input = """<|user|>
{instruction}

<|assistant|>
{output}
"""
```

### 指定 LoRA 目标模块

```python
from config import Config

config = Config()
config.lora.target_modules = ["q_proj", "v_proj"]  # 仅微调 attention
# config.lora.target_modules = None  # 自动选择
```

### 自定义数据列名

```python
from config import Config

config = Config()
config.data.instruction_column = "question"
config.data.input_column = "context"
config.data.output_column = "answer"
```

## 下一步

完成本阶段后，你可以：

1. 🏗️ **完整工程** - 学习 YAML 配置、日志追踪、模型部署
2. 📚 **阅读文档** - 深入理解 LoRA 原理和调参技巧

## 模块设计思想

```
┌─────────────────────────────────────────────────────┐
│                   train.py                          │
│                  (统一入口)                          │
└─────────────────────┬───────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
   ┌────────┐   ┌──────────┐   ┌─────────┐
   │ config │   │   data   │   │  model  │
   │  .py   │   │   .py    │   │   .py   │
   └────────┘   └──────────┘   └─────────┘
        │             │             │
        └─────────────┼─────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
   ┌──────────┐             ┌──────────────┐
   │  lora.py │             │full_finetune │
   └──────────┘             └──────────────┘
```

**设计原则：**
- **单一职责** - 每个模块只做一件事
- **可复用** - 函数和类可以在其他项目中重用
- **可配置** - 所有超参数集中在 config.py
- **易扩展** - 添加新功能不需要修改现有代码
