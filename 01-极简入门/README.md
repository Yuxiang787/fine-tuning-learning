# 01 - 极简入门

> 从一个单文件脚本入手，理解 LoRA 微调是怎么把“通用大模型”变成“更会做某类任务的模型”的。

这个目录适合第一次接触 LLM fine-tuning 的同学。默认读者已经知道这些基础概念：

- 大语言模型本质上是在做 next-token prediction
- tokenizer 会把文本切成 token id
- 训练时会通过 loss 和梯度下降更新参数

你暂时不需要提前掌握：

- PEFT / LoRA 的数学细节
- Hugging Face Trainer 的完整生态
- 分布式训练、量化训练、RLHF

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
| `lora_finetune.py` | 单文件 LoRA 微调脚本，教学主线 |
| `inference.py` | 加载基础模型 + LoRA 适配器做推理 |
| `data.jsonl` | 示例训练数据（instruction / input / output） |
| `requirements.txt` | Python 依赖 |

## 先建立直觉：这个脚本到底在做什么

我们并不是“从零训练一个模型”，也不是“把 5 亿参数全部改写一遍”。

这个脚本做的是：

1. 先加载一个已经会通用语言能力的基础模型，比如 `Qwen/Qwen2.5-0.5B`
2. 准备少量监督样本，让模型看到“什么输入应该对应什么输出”
3. 冻结原始大模型的大部分参数
4. 只在少数线性层上挂一组很小的 LoRA 可训练参数
5. 用这些样本训练这组小参数，让模型更贴近我们的任务格式

可以把它想象成：

```text
预训练模型负责“通用语言能力”
+ LoRA 适配器负责“这次任务的偏好修正”
= 一个更贴近你数据分布的模型
```

所以 LoRA 微调的核心收益是：

- 训练参数更少
- 显存占用更低
- 训练更快
- 保存结果更轻量，通常只需要保存 adapter，而不是整套基座模型

## LoRA 的机制，用初学者能落地的方式理解

### 传统全量微调

全量微调会直接更新基础模型中的大量甚至全部参数。

优点：

- 表达能力强
- 适合大数据、强适配需求

缺点：

- 显存和存储成本高
- 训练慢
- 每个任务都要保存一整份模型

### LoRA 微调

LoRA（Low-Rank Adaptation）不直接大改原始权重，而是在某些线性层旁边增加一个“低秩增量”：

```text
原始输出 = W x
LoRA 后输出 = W x + ΔW x

其中 ΔW 不直接学成一个完整大矩阵，
而是拆成两个更小的矩阵 A 和 B：
ΔW = B A
```

如果原始矩阵很大，而 rank `r` 很小，那么新增参数量就会远小于直接训练整个 `W`。

在工程上，你可以先记住这几点：

- `W` 通常被冻结，不更新
- 训练时只更新 LoRA 新增的小矩阵
- 推理时效果等价于“基础模型 + 任务适配器”
- 适配器可以单独保存、单独加载

### `r`、`alpha`、`dropout` 分别是什么

在这个项目里，LoRA 相关超参数在 [lora_finetune.py](/Users/ting/Documents/code/fine-tuning/01-极简入门/lora_finetune.py) 里定义：

```python
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
```

可以先这样理解：

- `r`：低秩分解的维度，越大通常容量越强，但参数和显存也会增加
- `lora_alpha`：LoRA 增量的缩放强度，常见经验是设为 `2 * r`
- `lora_dropout`：对 LoRA 分支做 dropout，帮助减少小数据过拟合

## 训练数据长什么样

训练样本在 [data.jsonl](/Users/ting/Documents/code/fine-tuning/01-极简入门/data.jsonl)，每行一条 JSON：

```json
{"instruction": "将以下中文翻译成英文", "input": "你好，世界！", "output": "Hello, world!"}
```

这个格式其实是在表达一件事：

- `instruction`：任务要求
- `input`：待处理内容
- `output`：期望答案

脚本会先把结构化字段拼成一段纯文本 prompt，例如：

```text
### Instruction:
将以下中文翻译成英文

### Input:
你好，世界！

### Output:
Hello, world!
```

这一步很重要，因为 causal LM 并不直接理解“字段名”，它真正看到的是一串 token。所谓 SFT，本质上就是让模型在这种 prompt 格式上继续学习“接下来应该输出什么”。

## 按代码顺序看完整训练流程

下面按 [lora_finetune.py](/Users/ting/Documents/code/fine-tuning/01-极简入门/lora_finetune.py) 的执行顺序来解释。

### 1. 选择设备

脚本会优先选择：

- Apple Silicon 的 `mps`
- NVIDIA GPU 的 `cuda`
- 否则退回 `cpu`

对应代码在 [lora_finetune.py](/Users/ting/Documents/code/fine-tuning/01-极简入门/lora_finetune.py) 中的设备检查部分。

这一步影响两件事：

- 模型和张量放在哪个设备上
- 是否启用半精度 `float16`

### 2. 读取 JSONL 数据

`load_data()` 会逐行读取 `data.jsonl`，返回 Python 列表。

这里仍然只是原始结构化样本，还没有变成 token。

### 3. 把样本格式化成训练文本

`format_prompt()` 会把：

- `instruction`
- `input`
- `output`

拼接成一段标准化 prompt 文本。

这一步的意义是统一训练模板。模型以后推理时也最好沿用同样的格式，否则训练分布和推理分布会不一致。

### 4. 加载 tokenizer

核心 API：

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
```

这里用的是 `transformers` 的自动类。`AutoTokenizer` 会根据模型名加载匹配的 tokenizer 配置。

脚本里还有一个很常见的小处理：

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

为什么这么做？

- 某些 causal LM 默认没有独立的 `pad_token`
- 但我们这里用了 `padding="max_length"`，所以需要一个 pad token
- 教学示例里把 `eos_token` 兼作 pad token，是非常常见的简化做法

### 5. 加载基础模型

核心 API：

```python
model = AutoModelForCausalLM.from_pretrained(...)
```

这里的 `CausalLM` 表示这是一个“自回归语言模型”，训练目标是根据前文预测后文。

代码里还用到了两个很常见的参数：

- `torch_dtype=...`
  作用：控制模型参数用什么精度加载
- `device_map="auto"`
  作用：让 `transformers/accelerate` 自动决定模型怎么放到设备上

在这个教学脚本里，我们训练的是一个已经预训练好的模型，不是随机初始化模型。

### 6. 配置 LoRA

核心 API：

```python
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

这里来自 `peft`：

- `LoraConfig`：定义 LoRA 应该怎么挂
- `TaskType.CAUSAL_LM`：告诉 PEFT 当前任务是自回归语言模型

其中 `target_modules=None` 是这个示例里一个很适合教学的点。

它表示：

- 不手动指定要改哪些层
- 让 `peft` 根据模型类型自动选择常见目标模块

对于新手，这是个很好的默认值，因为可以先专注理解流程，而不是一开始就钻进不同模型架构的模块命名差异。

### 7. 把 LoRA 挂到基础模型上

核心 API：

```python
model = get_peft_model(model, lora_config)
```

这一步之后：

- 原始模型大部分参数会被冻结
- 指定模块上会插入 LoRA adapter
- 只有 adapter 参数会参与训练

脚本随后统计：

- 总参数量
- 可训练参数量
- 可训练参数占比

这正是 LoRA 最值得观察的指标之一。你通常会看到只训练了不到 1% 的参数。

### 8. 用 `datasets.Dataset` 管数据

核心 API：

```python
dataset = Dataset.from_list(raw_data)
dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)
```

这里来自 `datasets` 库。

为什么不用普通 Python list 直接喂给 Trainer？

- `Dataset` 更适合做批处理和 map 变换
- 接口和 Hugging Face 训练生态兼容
- 以后扩展到 train/validation split、多进程预处理会更自然

这里的 `map()` 做了两件事：

1. 对每条样本执行 `format_prompt`
2. 删除旧字段，只保留新的 `text`

### 9. 分词，把文本变成 token id

脚本中的 `tokenize()` 调用：

```python
tokenizer(
    example["text"],
    truncation=True,
    max_length=MAX_LENGTH,
    padding="max_length",
)
```

这一步输出的通常包括：

- `input_ids`
- `attention_mask`

理解这几个参数很关键：

- `truncation=True`
  太长就截断，防止超过模型上下文长度或显存预算
- `max_length=MAX_LENGTH`
  统一最大长度
- `padding="max_length"`
  不足的样本补齐到同一长度，便于组成 batch

这一步之后，数据才真正变成“模型能计算的数值张量”。

### 10. 配置训练超参数

核心 API：

```python
training_args = TrainingArguments(...)
```

它来自 `transformers`，是 `Trainer` 的统一训练配置入口。

这个脚本里最值得关注的参数有：

- `num_train_epochs`
  训练轮数
- `per_device_train_batch_size`
  每张设备上的 batch size
- `learning_rate`
  学习率
- `logging_steps`
  每多少步打印一次日志
- `save_strategy="epoch"`
  每个 epoch 保存一次
- `fp16=device.type in ["cuda", "mps"]`
  在支持的设备上用半精度训练
- `gradient_accumulation_steps=2`
  用梯度累积模拟更大的总 batch
- `warmup_steps=10`
  训练前期做学习率预热

### 11. Data Collator 在这里做什么

核心 API：

```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)
```

这个类的职责是把多条样本拼成 batch，并准备训练所需字段。

这里 `mlm=False` 非常关键，表示：

- 我们不是 BERT 那种 masked language modeling
- 而是在做 causal language modeling

在这个例子里，它会基于输入构造语言模型训练所需的 `labels`。对于 causal LM，通常就是让模型学习预测下一个 token。

教学上要注意一点：

- 这个脚本是“极简版本”
- 它没有把 prompt 部分从 loss 里显式 mask 掉
- 也就是说，模型会同时学习整段文本，而不是只对 `output` 部分计算监督

这对入门理解流程很有帮助，但不是最严格的 SFT 写法。更完整的项目里，通常会对 prompt token 做 label mask，只让 answer 部分参与 loss。

### 12. Trainer 如何把这些东西串起来

核心 API：

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)
```

`Trainer` 帮我们封装了很多训练细节：

- dataloader 构建
- 前向计算
- loss 计算
- 反向传播
- optimizer step
- 日志打印
- checkpoint 保存

所以这份脚本才可以用很少的代码把完整流程跑起来。

### 13. `trainer.train()` 发生了什么

当执行：

```python
trainer.train()
```

底层大致会重复以下循环：

1. 取一个 batch 的 tokenized 样本
2. 前向传播，得到 logits 和 loss
3. 反向传播，把梯度传到 LoRA 参数
4. 更新可训练参数
5. 继续下一个 batch，直到一个 epoch 完成

因为基础模型主体被冻结，所以优化器主要只会更新 LoRA adapter 参数。

### 14. 保存的到底是什么

训练结束后：

```python
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
```

对于 PEFT 模型，这里保存的重点通常是：

- adapter 权重
- adapter 配置
- tokenizer 配置

而不是完整基础模型参数。

这也是为什么 LoRA 的产物通常只有几十 MB，而不是几 GB。

## 训练过程中模型“学到”的是什么

从优化角度看，这个脚本在最小化一件事：

- 给定 prompt 前缀
- 让模型更倾向于生成训练样本中的目标输出

举个例子，假设训练文本是：

```text
### Instruction:
判断以下句子的情感倾向

### Input:
这部电影太精彩了！

### Output:
正面
```

训练时模型会逐 token 地学习：

- 在看到 `### Instruction:` 后常出现任务描述
- 在看到 `### Output:` 后，应该生成与该任务匹配的答案
- 类似输入和类似任务描述会对应类似的输出模式

所以微调不是“让模型背题”这么简单，它也在学习：

- 指令格式
- 回答风格
- 特定任务与输出之间的映射关系

只是因为这个示例数据量很小，所以它更适合作为“理解流程”和“验证端到端打通”的教学案例，而不是追求强泛化效果。

## 这个脚本涉及到的核心库和 API

### `transformers`

主要用到：

- `AutoTokenizer.from_pretrained()`
- `AutoModelForCausalLM.from_pretrained()`
- `TrainingArguments`
- `Trainer`
- `DataCollatorForLanguageModeling`

职责可以概括为：

- 加载模型和 tokenizer
- 定义训练参数
- 提供通用训练循环

### `datasets`

主要用到：

- `Dataset.from_list()`
- `Dataset.map()`

职责：

- 把原始样本转换成适合训练的数据集对象
- 完成 prompt 格式化和 tokenize 预处理

### `peft`

主要用到：

- `LoraConfig`
- `get_peft_model()`
- `TaskType`

职责：

- 定义 LoRA 微调策略
- 把 LoRA adapter 注入基础模型
- 只训练少量新增参数

## 一个更贴近实际的心智模型

如果把整个训练过程抽象成一条数据流，可以理解为：

```text
JSONL 样本
-> prompt 文本
-> tokenizer 转成 token ids
-> model 前向计算 logits
-> 和目标 token 计算 loss
-> 反向传播
-> 更新 LoRA adapter 参数
-> 保存 adapter
```

这条链路就是你之后学习更复杂微调方法时反复会遇到的主线。以后即使换成：

- 更大的模型
- 更复杂的数据模板
- SFT label mask
- QLoRA / 4bit
- 多卡训练

本质上仍然是在这条主线上做增强。

## 自定义配置

### 更换模型

编辑 [lora_finetune.py](/Users/ting/Documents/code/fine-tuning/01-极简入门/lora_finetune.py) 中的 `MODEL_NAME`：

```python
MODEL_NAME = "Qwen/Qwen2.5-0.5B"        # 默认推荐，中文友好
MODEL_NAME = "facebook/opt-125m"        # 更小，适合快速验证流程
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 更大一些
```

### 调整 LoRA 参数

```python
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
```

经验上可以这样试：

- 显存紧张时，先减小 `r`
- 数据很少时，适当保守一点，避免过拟合
- 如果效果不够，再尝试增大 `r` 或增加数据量

### 调整训练参数

```python
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
MAX_LENGTH = 512
```

可以先用下面的直觉：

- `BATCH_SIZE` 越大，吞吐通常越高，但更吃显存
- `NUM_EPOCHS` 越大，训练更充分，但小数据更容易过拟合
- `LEARNING_RATE` 过大容易不稳定，过小又学不动
- `MAX_LENGTH` 决定单样本最大 token 长度，对显存影响明显

## 推理时为什么还需要基础模型

[inference.py](/Users/ting/Documents/code/fine-tuning/01-极简入门/inference.py) 的流程是：

1. 先加载基础模型
2. 再通过 `PeftModel.from_pretrained()` 挂载 `lora_output` 中的 adapter
3. 按训练时相同的 prompt 模板构造输入
4. 调用 `model.generate()` 生成答案

这是因为当前训练脚本默认保存的是 adapter，而不是一份完整合并后的大模型。

所以 LoRA 推理通常是：

```text
base model + adapter -> 推理
```

## 预期输出

运行训练脚本时，你会看到类似日志：

```text
============================================================
大模型 LoRA 微调教学示例
============================================================

[1] 使用设备：cuda / mps / cpu
[2] 加载数据：.../data.jsonl
[3] 加载模型：Qwen/Qwen2.5-0.5B
[4] 配置 LoRA 参数
[5] 处理数据集
[6] 配置训练参数
[7] 创建 Trainer
[8] 开始训练
```

训练过程中会周期性打印 `loss`。对于这个教学样例：

- `loss` 下降通常说明模型正在适配这批样本
- 但因为数据量很小，`loss` 不代表真实泛化能力

## 常见问题

### Q: CPU 能跑吗？

可以，但很慢。这个项目更适合：

- Apple Silicon + MPS
- NVIDIA GPU
- Colab / Kaggle 等带 GPU 的环境

### Q: 显存不足怎么办？

优先尝试：

1. 降低 `BATCH_SIZE`
2. 降低 `MAX_LENGTH`
3. 增大 `gradient_accumulation_steps`
4. 改用更小的基座模型

### Q: 为什么这里只有 20 条数据？

因为这个目录的目标是教学，不是追求效果上限。

少量数据足够帮助你观察：

- 数据如何格式化
- tokenization 怎么做
- LoRA 如何挂接
- Trainer 如何启动训练

如果想要更像真实项目的效果，通常需要更多、更干净、更一致的数据。

### Q: 这算不算标准 SFT？

算是一个极简版 SFT 示例，但不是最严格的工业写法。

原因是这个脚本没有显式对 prompt 部分做 label mask，而是直接把整段文本送进 causal LM 训练。教学上这样更容易理解，工程上则常常会进一步精细化。

### Q: 如何替换成自己的数据？

只要保持 `instruction / input / output` 这三个字段即可，例如：

```json
{"instruction": "解释以下概念", "input": "什么是过拟合？", "output": "过拟合是指模型在训练集上表现很好，但在新数据上泛化较差。"}
```

## 下一步学什么

如果你已经看懂这个目录，建议继续：

1. 看 [02-模块化项目/README.md](/Users/ting/Documents/code/fine-tuning/02-模块化项目/README.md)，理解如何把单文件脚本拆成更清晰的工程结构
2. 对比“只训练 adapter”和“全量更新模型参数”的区别
3. 进一步学习带 label mask 的 SFT、验证集评估、QLoRA 和更规范的数据处理

## 参考资源

- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [PEFT 文档](https://huggingface.co/docs/peft)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [Datasets 文档](https://huggingface.co/docs/datasets)
