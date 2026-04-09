# 03 - 完整工程

> 企业级微调项目结构 - YAML 配置、模块化设计、完整评测

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行训练

```bash
# 使用默认配置（LoRA 微调，默认记录到 TensorBoard）
python run.py

# 使用配置文件
python run.py --config configs/lora_config.yaml

# 命令行覆盖配置
python run.py --model_name Qwen/Qwen2.5-0.5B --num_epochs 5 --batch_size 8

# 全量微调
python run.py --config configs/full_finetune_config.yaml
```

### 3. 评测模型

```bash
# 评测 LoRA 模型
python eval.py --adapter ./output/lora_qwen0.5b

# 评测全量微调模型
python eval.py --model ./output/full_qwen0.5b

# 指定评测数据
python eval.py --adapter ./output --data eval_data.jsonl --num_samples 20
```

### 4. 推理交互

```bash
# 交互模式
python inference.py --adapter ./output/lora_qwen0.5b --interactive

# 演示模式
python inference.py --adapter ./output --demo
```

## API Usage In This Project

This section explains how different APIs are used inside the codebase itself.

### 1. `transformers` API

The project uses Hugging Face `transformers` as the main model API layer.

- `AutoTokenizer.from_pretrained(...)`
  Loads the tokenizer in `src/models/loader.py`.
- `AutoModelForCausalLM.from_pretrained(...)`
  Loads the base causal language model in `src/models/loader.py`.
- `TrainingArguments`
  Defines the training configuration in `src/train/trainer.py`, including batch size, learning rate, precision, checkpoint saving, and logging backend.
- `Trainer`
  Runs the training loop in `src/train/trainer.py`.
- `model.generate(...)`
  Is used in `src/eval/evaluator.py` for inference and evaluation.

So the project uses `transformers` for:

```text
load model/tokenizer -> configure training -> train -> save -> reload -> generate
```

### 2. Hugging Face Hub API

When the project uses a model name like `Qwen/Qwen2.5-0.5B`, it indirectly uses the Hugging Face Hub API through `from_pretrained(...)`.

That means the Hub is used to fetch:

- model config
- tokenizer config
- model weights
- generation config

The code does not manually call HTTP endpoints. The network access happens through the Hugging Face client inside `transformers` and `huggingface_hub`.

### 3. `peft` API

The project uses the `peft` library for LoRA training.

- `LoraConfig`
  Built in `src/models/lora_utils.py` to describe LoRA settings such as rank, alpha, and dropout.
- `get_peft_model(...)`
  Wraps the base model with LoRA adapters.
- `PeftModel.from_pretrained(...)`
  Reloads a trained LoRA adapter for evaluation or inference.

So the LoRA path is:

```text
base model -> attach LoRA adapters -> train adapters -> save adapters -> reload adapters
```

### 4. `datasets` API

The project uses Hugging Face `datasets` for preprocessing training data.

- `Dataset.from_list(...)`
  Converts JSONL records into a Hugging Face dataset object in `src/data/dataset.py`.
- `dataset.map(...)`
  Is used twice in `src/data/dataset.py`:
  - first to format raw examples into prompt text
  - then to tokenize the prompt text

This is the data-processing API layer between raw JSONL and model-ready inputs.

### 5. TensorBoard API

TensorBoard is the default experiment logging backend.

- The trainer enables TensorBoard by passing `report_to="tensorboard"` into `TrainingArguments`.
- The project sets `TENSORBOARD_LOGGING_DIR` in `src/train/trainer.py`.
- During training, the `Trainer` integration writes event files under `output/logs/<run_name>/`.

So TensorBoard is not called through custom scalar-writing code in this project; it is used through the `transformers` training integration.

### 6. Weights & Biases API

W&B support is optional.

- The project does not directly call `wandb.init()` or `wandb.log()`.
- Instead, it enables W&B through `TrainingArguments(report_to="wandb")`.
- Once enabled, the `Trainer` integration inside `transformers` sends metrics to W&B.

So W&B is used through the `transformers` integration API, not through handwritten W&B client calls.

### 7. PyTorch API

PyTorch is the execution backend under everything else.

The project uses PyTorch for:

- device selection (`cpu`, `cuda`, `mps`)
- dtype selection (`float16`, `float32`)
- tensor execution during forward and backward passes
- gradient computation
- optimizer updates handled by `Trainer`

Most of the training loop is abstracted by `transformers.Trainer`, but the project still makes explicit PyTorch-level choices about device and dtype in the loader and trainer modules.

## 项目结构

```
03-完整工程/
├── configs/                    # YAML 配置文件
│   ├── lora_config.yaml        # LoRA 微调配置
│   └── full_finetune_config.yaml # 全量微调配置
├── src/                        # 源代码
│   ├── __init__.py
│   ├── data/                   # 数据模块
│   │   ├── __init__.py
│   │   ├── dataset.py          # 数据集类
│   │   └── collator.py         # 数据收集器
│   ├── models/                 # 模型模块
│   │   ├── __init__.py
│   │   ├── loader.py           # 模型加载
│   │   └── lora_utils.py       # LoRA 工具
│   ├── train/                  # 训练模块
│   │   ├── __init__.py
│   │   ├── args.py             # 参数解析
│   │   └── trainer.py          # 训练器
│   └── eval/                   # 评测模块
│       ├── __init__.py
│       └── evaluator.py        # 评测器
├── scripts/                    # 工具脚本
├── logs/                       # 日志目录
├── run.py                      # 训练入口
├── eval.py                     # 评测入口
├── inference.py                # 推理入口
├── data.jsonl                  # 示例数据
└── requirements.txt            # 依赖
```

## 配置文件说明

### LoRA 配置 (configs/lora_config.yaml)

```yaml
# 模型配置
model_name: "Qwen/Qwen2.5-0.5B"

# 数据配置
data: "data.jsonl"
max_length: 512

# LoRA 配置
use_lora: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.1

# 训练配置
output_dir: "./output/lora_qwen0.5b"
num_epochs: 3
batch_size: 8
learning_rate: 2e-4

# 性能配置
fp16: true
```

### 全量微调配置 (configs/full_finetune_config.yaml)

```yaml
use_lora: false
full_finetune: true

# 全量微调需要更小的学习率
learning_rate: 1e-5
batch_size: 1
gradient_accumulation_steps: 8
```

## 命令行参数

### run.py (训练)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | - |
| `--model_name` | 模型名称 | Qwen/Qwen2.5-0.5B |
| `--data` | 训练数据 | data.jsonl |
| `--output_dir` | 输出目录 | ./output |
| `--num_epochs` | 训练轮数 | 3 |
| `--batch_size` | 批次大小 | 8 |
| `--learning_rate` | 学习率 | 2e-4 |
| `--lora_r` | LoRA 秩 | 16 |
| `--lora_alpha` | LoRA alpha | 32 |
| `--full_finetune` | 全量微调 | 否 |
| `--fp16` | FP16 混合精度 | 是 |
| `--report_to` | 日志工具 | tensorboard |

### eval.py (评测)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--base_model` | 基础模型 | Qwen/Qwen2.5-0.5B |
| `--adapter` | LoRA 适配器路径 | - |
| `--model` | 微调模型路径 | - |
| `--data` | 评测数据 | data.jsonl |
| `--num_samples` | 评测样本数 | 10 |
| `--output` | 结果保存路径 | - |

### inference.py (推理)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--base_model` | 基础模型 | Qwen/Qwen2.5-0.5B |
| `--adapter` | LoRA 适配器路径 | - |
| `--model` | 微调模型路径 | - |
| `--demo` | 演示模式 | 否 |
| `--interactive` | 交互模式 | 是 |

## 模块化设计

### 数据模块 (src/data/)

```python
from src.data import FineTuningDataset, load_dataset

# 方式 1: 使用类
dataset = FineTuningDataset(
    data_path="data.jsonl",
    format_template="...",
    format_template_no_input="..."
)
dataset.load().format().tokenize(tokenizer, max_length=512)
hf_dataset = dataset.to_huggingface()

# 方式 2: 使用便捷函数
dataset = load_dataset(
    data_path="data.jsonl",
    format_template="...",
    tokenizer=tokenizer
)
```

### 模型模块 (src/models/)

```python
from src.models import load_model, load_tokenizer, apply_lora, create_lora_config

# 加载
tokenizer = load_tokenizer("Qwen/Qwen2.5-0.5B")
model, device = load_model("Qwen/Qwen2.5-0.5B")

# 应用 LoRA
lora_config = create_lora_config(r=8, alpha=16)
model, trainable, total = apply_lora(model, lora_config)
```

### 训练模块 (src/train/)

```python
from src.train import create_trainer, train

# 使用内置训练函数
train(config_dict)

# 或自定义
trainer = create_trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    training_args=training_args
)
trainer.train()
```

### 评测模块 (src/eval/)

```python
from src.eval import Evaluator, load_eval_data

# 加载模型
evaluator = Evaluator.from_lora("Qwen/Qwen2.5-0.5B", "./output")

# 评测
data = load_eval_data("eval.jsonl")
results = evaluator.evaluate_batch(data)
evaluator.print_results(results)
```

## 日志与监控

### TensorBoard

TensorBoard 是当前默认日志后端，因此直接运行 `python run.py` 就会写入日志。
项目会自动把日志写到 `output/logs/<run_name>/`；如果未指定 `run_name`，会使用默认名称。

```bash
# 默认训练（自动写入 TensorBoard）
python run.py --run_name my_experiment

# 也可以显式指定
python run.py --report_to tensorboard --run_name my_experiment

# 查看所有运行
tensorboard --logdir ./output/logs
```

### Weights & Biases

```bash
# 登录
wandb login

# 如需改用 W&B
python run.py --report_to wandb --run_name my_experiment
```

## 输出目录结构

```
output/
└── lora_qwen0.5b/
    ├── adapter_config.json       # LoRA 配置
    ├── adapter_model.safetensors # LoRA 权重
    ├── tokenizer.json            # 分词器
    ├── tokenizer_config.json
    ├── checkpoint-500/           # 训练检查点
    │   ├── adapter_model.safetensors
    │   └── trainer_state.json
    └── logs/                     # 训练日志
        └── events.out.tfevents.*
```

## 自定义扩展

### 添加新模型

```python
# src/models/loader.py 已支持任意 HuggingFace 模型
model, _ = load_model("Qwen/Qwen2.5-0.5B", device)
```

### 添加数据增强

```python
# src/data/dataset.py
class AugmentedDataset(FineTuningDataset):
    def augment(self):
        # 添加数据增强逻辑
        return self
```

### 添加自定义评估指标

```python
# src/eval/evaluator.py
class CustomEvaluator(Evaluator):
    def compute_bleu(self, results):
        # 添加 BLEU 分数计算
        pass
```

## 最佳实践

### 1. 配置管理

- 使用 YAML 文件管理实验配置
- 命令行参数覆盖配置文件
- 保存配置到输出目录便于复现

### 2. 数据准备

- 训练前划分验证集（10-20%）
- 确保数据格式一致
- 检查数据质量和多样性

### 3. 训练调优

- LoRA 从 r=8, alpha=16 开始
- 全量微调学习率设为 LoRA 的 1/10
- 使用梯度累积模拟大批次

### 4. 评测

- 使用独立测试集
- 记录多个指标（精确匹配、关键词匹配）
- 保存详细结果便于分析

## 下一步

完成本阶段后，你可以：

1. 📚 **阅读文档** - 深入理解原理和调参
2. 🚀 **实际应用** - 使用自己的数据进行微调
3. 🔧 **扩展功能** - 添加自定义模块

## 项目对比

| 特性 | 01-极简 | 02-模块化 | 03-完整工程 |
|------|--------|----------|-------------|
| 单文件 | ✓ | - | - |
| 模块化 | - | ✓ | ✓ |
| YAML 配置 | - | - | ✓ |
| 命令行完整参数 | - | 部分 | ✓ |
| 日志追踪 | - | - | ✓ |
| 评测系统 | 基础 | 中等 | 完整 |
| 适用场景 | 入门 | 学习 | 生产 |
