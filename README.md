# 大模型微调教学项目

> 从零开始，循序渐进学习大语言模型微调

本项目是一个**面向教学**的大模型微调教程，通过三个阶段帮助你从入门到精通：

- 📦 **01-极简入门** - 单文件脚本，10 分钟跑通第一个微调
- 🔧 **02-模块化项目** - 清晰的项目结构，支持 LoRA 和全量微调
- 🏗️ **03-完整工程** - YAML 配置、完整评测、生产就绪

## 快速开始

### 1. 安装依赖

```bash
# 推荐 Python 3.11（最低 3.10）
pip install -r requirements.txt
```

### 2. 选择学习路径

| 阶段 | 适合人群 | 时间 | 输出 |
|------|----------|------|------|
| [01-极简入门](01-极简入门) | 第一次接触微调 | 30 分钟 | 单文件脚本 |
| [02-模块化项目](02-模块化项目) | 想理解内部原理 | 2 小时 | 模块化代码 |
| [03-完整工程](03-完整工程) | 想做实际应用 | 1 天 | 完整项目 |

### 3. 开始学习

```bash
# 从极简入门开始
cd 01-极简入门
python lora_finetune.py
```

## 项目结构

```
fine-tuning/
├── 01-极简入门/              # 第一阶段：单文件示例
│   ├── lora_finetune.py      # LoRA 微调脚本
│   ├── inference.py          # 推理脚本
│   ├── data.jsonl            # 示例数据
│   └── README.md
│
├── 02-模块化项目/            # 第二阶段：模块化设计
│   ├── config.py             # 配置管理
│   ├── data.py               # 数据处理
│   ├── model.py              # 模型加载
│   ├── lora.py               # LoRA 微调
│   ├── full_finetune.py      # 全量微调
│   ├── train.py              # 训练入口
│   ├── evaluate.py           # 评测脚本
│   └── README.md
│
├── 03-完整工程/              # 第三阶段：完整工程
│   ├── configs/              # YAML 配置
│   ├── src/                  # 源代码
│   │   ├── data/             # 数据模块
│   │   ├── models/           # 模型模块
│   │   ├── train/            # 训练模块
│   │   └── eval/             # 评测模块
│   ├── run.py                # 训练入口
│   ├── eval.py               # 评测入口
│   ├── inference.py          # 推理入口
│   └── README.md
│
├── docs/                     # 文档
│   ├── 01-lora 原理.md        # LoRA 原理解析
│   ├── 02-数据准备.md         # 数据准备指南
│   ├── 03-超参数调优.md       # 调参技巧
│   └── 04-常见问题.md         # FAQ
│
└── requirements.txt          # 全局依赖
```

## 学习路线

### 第一阶段：极简入门 (30 分钟)

**目标**: 快速跑通第一个微调，建立信心。

**内容**:
- 理解 LoRA 基本概念
- 运行单文件微调脚本
- 测试微调后的模型

**完成后你将理解**:
- 什么是 LoRA 微调
- 训练数据格式
- 如何保存和加载模型

### 第二阶段：模块化项目 (2 小时)

**目标**: 理解微调的完整流程和模块设计。

**内容**:
- 配置管理（config.py）
- 数据处理（data.py）
- 模型加载（model.py）
- LoRA 与全量微调对比
- 评测方法

**完成后你将理解**:
- 微调的完整流程
- 各模块的职责
- 如何选择 LoRA 或全量微调

### 第三阶段：完整工程 (1 天)

**目标**: 掌握生产级微调项目的设计。

**内容**:
- YAML 配置管理
- 模块化源码结构
- 训练日志与监控
- 完整评测系统
- 推理部署

**完成后你将能够**:
- 设计微调项目架构
- 管理实验配置
- 评估模型效果
- 部署到生产环境

## 核心概念

### 什么是 LoRA？

LoRA（Low-Rank Adaptation）是一种**参数高效微调**技术：

```
传统微调：████████████████████ 100% 参数可训练
LoRA 微调：████████████████████ 基础参数冻结
           └── ▒▒ 低秩适配器 仅 0.1-1% 参数可训练
```

**优势**:
- 显存占用降低 60-80%
- 训练速度提升 2-3 倍
- 保存大小从 GB 降到 MB

### 微调流程

```
1. 准备数据 → 2. 加载模型 → 3. 配置 LoRA → 4. 训练 → 5. 评测 → 6. 部署
     ↓              ↓             ↓            ↓         ↓         ↓
  JSONL 文件   HuggingFace   低秩适配器   Trainer   测试集   推理脚本
```

## 推荐模型

| 模型 | 大小 | 显存需求 | 中文支持 | 推荐场景 |
|------|------|----------|----------|----------|
| [OPT-125M](https://huggingface.co/facebook/opt-125m) | 125M | 1GB | 弱 | 学习、测试 |
| [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) | 0.5B | 2GB | 强 | 中文任务 |
| [TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | 1.1B | 3GB | 中 | 进阶使用 |

## 硬件需求

| 配置 | 最低 | 推荐 |
|------|------|------|
| GPU | CPU（慢） | Apple Silicon MPS / NVIDIA 4GB+ |
| 内存 | 8GB | 16GB |
| 存储 | 10GB | 20GB SSD |

**Apple Silicon**: M1/M2/M3/M4 芯片利用 MPS (Metal Performance Shaders) 加速，24GB 内存可运行 0.5B-1B 模型

**无 GPU 用户**: 使用 Google Colab 或 Kaggle Notebook（免费 T4 GPU）

## 文档导航

- 📖 [LoRA 原理](docs/01-lora 原理.md) - 深入理解 LoRA 工作原理
- 📊 [数据准备](docs/02-数据准备.md) - 如何准备训练数据
- 🎛️ [超参数调优](docs/03-超参数调优.md) - 调参技巧和最佳实践
- ❓ [常见问题](docs/04-常见问题.md) - FAQ 和故障排除

## 学习检查清单

完成学习后，你应该能够：

- [ ] 解释 LoRA 的基本原理
- [ ] 准备指令微调数据
- [ ] 运行 LoRA 微调训练
- [ ] 评估微调后的模型
- [ ] 选择合适的超参数
- [ ] 诊断常见问题

## 下一步

完成本教程后，你可以：

1. 📚 阅读 [LoRA 原论文](https://arxiv.org/abs/2106.09685)
2. 🔧 尝试 [QLoRA](https://arxiv.org/abs/2305.14314) 量化微调
3. 🏗️ 学习完整项目部署（vLLM、TGI 等）
4. 🚀 参与开源项目或实际应用

## 参考资源

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PEFT 文档](https://huggingface.co/docs/peft)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)

## 许可证

MIT License
