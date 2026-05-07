# 基于端云协同的大语言模型智能体优化方法

本仓库是本科毕业设计项目的代码与实验材料，主题为 **端云协同场景下的大语言模型智能体推理优化**。项目围绕 ReAct 与 LLMCompiler 两类智能体架构，比较云端大模型、边缘小模型和端云协同策略在 WildBench 与 LMSYS-Chat-1M 数据集上的质量、延迟、token 成本、通信开销和工具调用开销。

项目主要包含：

- `ours/`：本文方法，实现分层经验检索、工具记忆库和失败反思更新。
- `AWM/`、`CE-CoLLM/`、`Ecoagent/`：对比基线实现与实验结果。
- `ours/vis/`：端云协同智能体执行过程可视化系统。
- `requirements.txt`：Python 依赖列表。

## 方法概览

传统 LLM Agent 在处理多步推理、搜索、计算和代码生成任务时，通常依赖云端大模型完成完整推理流程，容易带来较高的调用成本、通信开销和端到端延迟。纯边缘小模型虽然成本低，但复杂任务能力不足。

本项目的核心思想是：优先让边缘小模型利用历史经验和工具记忆完成可复用任务；只有当经验不足、任务复杂或小模型执行失败时，再引入云端大模型。

本文方法由三个开关控制，位于各实验目录下的 `config.py`：

| 开关 | 作用 |
| --- | --- |
| `HIERARCHICAL_TRAJECTORY_RETRIEVAL` | 分层轨迹检索：高相似度返回历史执行轨迹，中等相似度返回抽象经验，低相似度回退云端大模型 |
| `TOOL_MEMORY_LIBRARY` | 工具记忆库：对工具输入进行相似度检索，命中后复用历史工具输出，减少真实 API 调用 |
| `FAILURE_EXPERIENCE_UPDATE` | 失败经验更新：当小模型使用经验后评分过低时，调用大模型生成反思并写回经验库 |

相关阈值也在 `config.py` 中配置：

- `SIMILARITY_THRESHOLD_1`：轨迹复用阈值。
- `SIMILARITY_THRESHOLD_2`：经验复用阈值。
- `TOOL_SIMILARITY_THRESHOLD`：工具记忆命中阈值。
- `FAILURE_SCORE_THRESHOLD`：触发失败反思的评分阈值。

## 目录结构

```text
Graduation_Design/
├── ours/
│   ├── WildBench/
│   │   └── React/              # 本文方法，WildBench，ReAct，8B 小模型
│   ├── WildBench_2B/
│   │   └── React/              # 本文方法，WildBench，ReAct，2B 小模型消融
│   ├── LMSYS_CHAT_1M/
│   │   ├── React/              # 本文方法，LMSYS-Chat-1M，ReAct
│   │   └── LLMCompiler/        # 本文方法，LMSYS-Chat-1M，LLMCompiler
│   └── vis/                    # 可视化系统
│       ├── backend/            # FastAPI + WebSocket 后端
│       ├── frontend/           # 静态前端页面
│       ├── agent_core/         # 可视化系统调用的 Agent 核心
│       └── data/               # 经验库、工具库、执行历史
│
├── AWM/                        # Agent Workflow Memory 基线
├── CE-CoLLM/                   # 端云协同 LLM 基线
├── Ecoagent/                   # EcoAgent 基线
├── React/                      # 已存在的 ReAct 虚拟环境目录
├── LLMcompiler/                # 已存在的 LLMCompiler 虚拟环境目录
├── testcode/                   # 调试和实验辅助脚本
├── requirements.txt
└── README.md
```

典型实验目录包含如下文件：

| 文件 | 说明 |
| --- | --- |
| `main.py` | 实验入口，负责加载数据、运行 Agent、保存结果和汇总指标 |
| `config.py` | API、模型、数据集、阈值和创新点开关配置 |
| `agent.py` | ReAct 或 LLMCompiler 智能体逻辑 |
| `llm_client.py` | OpenAI-compatible LLM 调用封装 |
| `tools.py` | 搜索、计算、日期、代码生成等工具实现 |
| `tool_db.py` | 工具记忆库，使用 FAISS 做相似度检索 |
| `experience_db.py` | 经验库、轨迹检索、经验写入和失败反思 |
| `embedding_utils.py` | Qwen Embedding 本地/API 封装 |
| `dataset_utils.py` | WildBench / LMSYS-Chat-1M 数据加载与任务拆分 |
| `evaluation.py` | LLM-as-a-Judge 评分模块 |

## 环境准备

推荐使用 Python 3.10 或更高版本。

```bash
cd /home/gujing/Graduation_Design
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果希望复用仓库中已有的虚拟环境，可以按实验类型激活：

```bash
source React/bin/activate
# 或
source LLMcompiler/bin/activate
```

## 配置环境变量

根目录使用 `.env` 管理密钥和服务地址。`.env` 已被 `.gitignore` 忽略，不要提交到仓库。

可参考以下模板：

```bash
# 云端大模型
DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_API_BASE=https://api.deepseek.com/v1

# 评测模型
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_API_BASE=https://openrouter.ai/api/v1

# 代码生成模型
PPINFRA_API_KEY=your_ppinfra_key
PPINFRA_API_BASE=https://api.ppinfra.com/openai

# 边缘小模型，要求兼容 OpenAI /v1/chat/completions
SMALL_MODEL_API_KEY=token-abc123
SMALL_MODEL_API_BASE=http://127.0.0.1:8000/v1

# Embedding 服务，要求兼容 OpenAI /v1/embeddings
SMALL_MODEL_EMBEDDING_BASE=http://127.0.0.1:8001/v1

# 工具 API
SERPER_API_KEY=your_serper_key
GOOGLE_CSE_API_KEY=your_google_cse_key
GOOGLE_CSE_CX=your_google_cse_id

# HuggingFace 数据集访问
HF_TOKEN=your_huggingface_token
```

小模型与嵌入模型可以使用本地 vLLM、Ollama 兼容服务或任意 OpenAI-compatible 服务。以 vLLM 为例：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --port 8000 \
  --api-key token-abc123

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-Embedding-0.6B \
  --port 8001 \
  --api-key token-abc123
```

## 运行本文方法

### WildBench + ReAct

```bash
cd /home/gujing/Graduation_Design/ours/WildBench/React
python main.py --sample_size 50 --no_eval
```

常用参数：

- `--sample_size 50`：指定本次加载的样本数量；不指定时使用 `config.py` 中的 `DATASET_LIMIT`。
- `--no_eval`：跳过 LLM-as-a-Judge 评测，适合先验证流程。
- `--mode hybrid`：当前优化方法只支持 hybrid，默认可省略。

### WildBench 2B 消融

```bash
cd /home/gujing/Graduation_Design/ours/WildBench_2B/React
python main.py --sample_size 50 --no_eval
```

该目录默认使用 `Qwen/Qwen3.5-2B` 小模型配置，用于比较不同边缘模型能力下的端云协同效果。

### LMSYS-Chat-1M + ReAct

```bash
cd /home/gujing/Graduation_Design/ours/LMSYS_CHAT_1M/React
python main.py --sample_size 50 --no_eval
```

不传 `--sample_size` 时，代码会按 `SAMPLE_SIZE_Multi` 和 `SAMPLE_SIZE_Single` 分别采样多轮与单轮英文对话。

### LMSYS-Chat-1M + LLMCompiler

```bash
cd /home/gujing/Graduation_Design/ours/LMSYS_CHAT_1M/LLMCompiler
python main.py --mode hybrid --sample_size 50 --no_eval
```

## 运行基线

### AWM

```bash
cd /home/gujing/Graduation_Design/AWM/WildBench/React
python main.py --sample_size 50 --no_eval

cd /home/gujing/Graduation_Design/AWM/LMSYS_CHAT_1M/React
python main.py --sample_size 50 --no_eval

cd /home/gujing/Graduation_Design/AWM/LMSYS_CHAT_1M/LLMCompiler
python main.py --mode hybrid --sample_size 50 --no_eval
```

### CE-CoLLM

`CE-CoLLM` 支持三种模式：`large_only`、`small_only`、`hybrid`。

```bash
cd /home/gujing/Graduation_Design/CE-CoLLM/WildBench
python main.py --mode large_only --sample_size 50 --no_eval
python main.py --mode small_only --sample_size 50 --no_eval
python main.py --mode hybrid --sample_size 50 --no_eval
```

LMSYS-Chat-1M 和 2B 实验目录同理：

```bash
cd /home/gujing/Graduation_Design/CE-CoLLM/LMSYS_CHAT_1M
python main.py --mode hybrid --sample_size 50 --no_eval

cd /home/gujing/Graduation_Design/CE-CoLLM/WildBench_2B
python main.py --mode hybrid --sample_size 50 --no_eval
```

### EcoAgent

```bash
cd /home/gujing/Graduation_Design/Ecoagent/LMSYS_CHAT_1M
python main.py --mode large_only --sample_size 50 --no_eval
python main.py --mode small_only --sample_size 50 --no_eval
python main.py --mode hybrid --sample_size 50 --no_eval
```

## 结果文件

实验会在当前运行目录下追加写入结果，支持断点续跑。重新运行时，代码会读取已有 `task_id` 并跳过已处理样本。

常见输出包括：

| 文件 | 说明 |
| --- | --- |
| `results.jsonl` | 本文方法或 AWM ReAct 的逐样本结果 |
| `results_hybrid.jsonl` | hybrid 模式逐样本结果 |
| `results_large_only.jsonl` | large-only 模式逐样本结果 |
| `results_small_only.jsonl` | small-only 模式逐样本结果 |
| `summary.json` | 平均指标汇总 |
| `summary_hybrid.json` | hybrid 模式平均指标汇总 |
| `experience_db.json` | 经验库元数据 |
| `experience_db.index` | FAISS 经验索引 |
| `edge_tool_db.json` / `cloud_tool_db.json` | 端侧/云侧工具记忆库 |

`*.index` 文件由 FAISS 生成。如果索引缺失或与 JSON 数量不一致，代码会根据 JSON 重新构建。

## 评测指标

项目主要记录以下指标：

| 类别 | 指标 |
| --- | --- |
| 质量 | `llm_judge_score`，0-10 分 |
| 延迟 | `latency_seconds` |
| 模型调用 | `large_model_calls`、`small_model_calls` |
| Token | `prompt_tokens`、`completion_tokens`、`total_tokens` |
| 工具 | `search_calls`、`code_tokens` |
| 通信 | `edge_to_cloud_kb`、`cloud_to_edge_kb` |

WildBench 实验会尽量利用数据集中的 `references` 和 `checklist` 进行评测；LMSYS 实验会基于对话上下文进行 LLM-as-a-Judge 评分。

## 可视化系统

可视化系统用于展示端云协同决策过程，包括：

- ReAct / LLMCompiler 架构切换。
- 分层检索、工具记忆、失败反思三个创新点开关。
- T1、T2 和工具相似度阈值配置。
- 实时执行 trace、端云服务位置、工具/经验库内容。
- LLM 成本、搜索次数、代码生成成本和延迟统计。
- 对话历史保存与回放。

启动方式：

```bash
cd /home/gujing/Graduation_Design/ours/vis
bash start.sh
```

启动后访问：

- 前端页面：http://127.0.0.1:8003
- 后端服务：http://127.0.0.1:8002

`start.sh` 会占用并清理本机 `8002` 和 `8003` 端口。

## 数据集

项目使用 HuggingFace 数据集：

- WildBench：`allenai/WildBench`，配置 `v2`，默认 split 为 `test`。
- LMSYS-Chat-1M：`lmsys/lmsys-chat-1m`，默认 split 为 `train`，代码会筛选英文对话。

首次运行会下载并缓存数据集。如果没有配置 `HF_TOKEN` 或网络访问失败，部分 `dataset_utils.py` 会返回少量 mock 数据用于流程调试；正式实验请确保 HuggingFace Token 和网络可用。

## 常见问题

### 1. 小模型接口连接失败

检查 `SMALL_MODEL_API_BASE` 是否指向可用的 OpenAI-compatible 服务，例如 `http://127.0.0.1:8000/v1`。同时确认 `SMALL_MODEL_API_KEY` 与服务启动时的 `--api-key` 一致。

### 2. Embedding 初始化失败

检查 `SMALL_MODEL_EMBEDDING_BASE` 是否可访问，并确认服务支持 `/v1/embeddings`。如果使用本地 SentenceTransformer，可在对应 `config.py` 中将 `EMBEDDING_USE_LOCAL` 改为 `True`。

### 3. 搜索工具不可用

`web_search` 默认依赖 `SERPER_API_KEY`。如果缺少密钥，Agent 会收到工具错误信息，实验仍可继续，但搜索类任务质量会下降。

### 4. 评测太慢或调用成本太高

调试阶段建议加 `--no_eval`；正式实验再开启评测。评测模型由 `OPENROUTER_API_KEY`、`OPENROUTER_API_BASE` 和 `EVALUATION_LLM_MODEL` 控制。

### 5. 想重新跑同一批实验

当前实验支持断点续跑，会跳过已有 `task_id`。如果需要完整重跑，请先备份或删除当前目录下对应的 `results*.jsonl` 和 `summary*.json`。

## 说明

本仓库用于毕业设计实验复现和学术研究展示。运行实验前请根据自己的模型服务、API 密钥和数据集权限修改对应目录下的 `config.py` 与根目录 `.env`。
