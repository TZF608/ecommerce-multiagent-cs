# 星火数码 · 电商智能客服 Agent

基于 **LangGraph + Agentic RAG + 工具调用 + 人工接管** 的电商智能客服多智能体系统（售前咨询 / 售后处理 / 投诉升级）。
用于简历项目的完整落地示例：明确业务场景、完整技术链路、可量化评估。

## 简历亮点（数字需用真实评测产出）

| 维度 | 内容 |
|---|---|
| 业务场景 | 电商售前/售后客服：商品咨询、订单查询、退换货、退款、发票、投诉升级 |
| 技术链路 | 数据构建 → RAG（jieba BM25 混合检索）→ LangGraph 多智能体 → 工具调用 → 人工接管 → LLM-as-judge 评测 → FastAPI 部署 |
| 可量化成果 | 检索 recall、意图准确率、答案通过率、无人工接管解决率、平均解决轮数、单会话成本 |

## 多智能体架构

```
user_input
    │
    ▼
[router 意图路由]  ──► chitchat ──► 兜底回复
    │
    ├─ presale   ──► [presale_agent]   Agentic RAG：search_knowledge_base 工具
    ├─ aftersale ──► [aftersale_agent] 订单工具(query_order/物流/退款) + 知识库
    └─ complaint ─► [upgrade_agent]    interrupt 人工接管，resume 后由人工接管
```

- **会话记忆**：`MemorySaver` 按 `thread_id` 保存整段对话
- **多轮检索**：指代类追问（"那它呢"）先由路由继承上一轮意图，再把 query 改写成语义完整的问句（LLM 结合对话历史）后检索；工具层同样内置改写兜底
- **人工接管**：投诉分支 `interrupt()` 挂起，API 提供 `/handoff/resume` 恢复
- **成本统计**：`UsageCallbackHandler` 汇总每次对话 token，按模型估算成本

## 目录结构

```
d:\project
├── data/
│   ├── raw/                  # 商品库(24)、售后规则、FAQ（含联系人工客服）
│   └── eval/
│       └── eval_queries.json # 100 条带标注评估集
├── src/
│   ├── config.py / llm.py    # 配置与 LLM/Embedding 工厂
│   ├── data/loader.py        # 原始数据 -> Document（规则/FAQ 按小节切分）
│   ├── rag/
│   │   ├── ingest.py         # 分块 -> 向量化 -> Chroma
│   │   ├── retriever.py      # jieba BM25 + 向量 混合检索(RRF) + 重排(可选)
│   │   ├── rewrite.py        # 多轮 query 改写：把"那款呢"补全成完整问句再检索
│   │   └── chain.py          # 纯 RAG 问答链（基线对比用）
│   ├── agent/
│   │   ├── graph.py          # LangGraph 状态机（核心）
│   │   ├── nodes.py          # 路由/售前/售后/升级/兜底节点
│   │   ├── tools.py          # @tool：订单查询/物流/退款 + 知识库检索
│   │   ├── prompts.py        # 各子 Agent 系统提示词
│   │   └── callbacks.py      # token 用量与成本估算
│   ├── eval/llm_judge.py     # LLM-as-judge 打分
│   └── api/
│       ├── main.py           # FastAPI：/chat、/handoff/resume
│       └── static/index.html # 聊天前端
└── scripts/
    ├── ingest.py             # 构建向量库
    ├── evaluate.py           # 检索评估（无需 LLM）
    ├── eval_intents.py       # 意图路由准确率（需 LLM）
    ├── eval_answers.py       # 端到端答案评测（需 LLM）
    ├── ablation.py           # 消融实验：纯向量 / 纯 BM25 / 混合对比
    ├── run_demo.py           # 纯 RAG 问答 demo
    └── chat.py               # 完整多智能体对话 CLI

tests/                      # pytest 单元测试（路由/检索/工具/中断恢复/评测集 schema）
```

## 快速开始

```powershell
# 1. 创建虚拟环境并激活
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. 安装依赖（推荐先装 CPU 版 torch，约 200MB，避免默认下载 2.5GB+ 的 CUDA 版）
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. 配置 LLM：复制 .env.example 为 .env，任选一家厂商填好
#    DeepSeek / Kimi / 通义 / 智谱 / OpenAI / Ollama 均可

# 4. 构建向量库（首次会下载中文 embedding 模型；连不上 huggingface 时先执行
#    $env:HF_ENDPOINT='https://hf-mirror.com'）
python scripts/ingest.py

# 5. 三条评估线（依次跑，数字沉淀进简历）
python scripts/evaluate.py        # 检索评估（无需 LLM）
python scripts/eval_intents.py    # 意图准确率（需 LLM）
python scripts/eval_answers.py    # 端到端答案评测（需 LLM）

# 6. 对话
python scripts/chat.py            # 完整多智能体对话（支持投诉转人工）
python scripts/run_demo.py        # 纯 RAG 基线 demo（对比用）
```

## Web 服务

```powershell
# 方式一（推荐）：双击项目根目录的 start_web.bat，自动启动并打开浏览器
# 方式二：命令行
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --port 8000
# 打开 http://127.0.0.1:8000 使用聊天界面
# 接口：POST /chat、POST /handoff/resume、GET /health
# 面试/演示话术见 docs/DEMO.md
```

## 常用命令备忘

| 命令 | 作用 |
|---|---|
| `python scripts/ingest.py` | 重建向量库（先清空旧的，无重复） |
| `python scripts/evaluate.py -k 4` | 检索评估（TOP_K=4） |
| `python scripts/run_demo.py -q "问题"` | 纯 RAG 单次问答 |
| `python scripts/chat.py` | 完整多智能体对话 |
| `python scripts/eval_answers.py` | 端到端答案评测 |
| `uvicorn src.api.main:app --reload` | 启动 Web 服务 |

## 自动化测试与消融实验

```powershell
# 单元测试（离线可跑，无需 LLM / embedding 模型）
python -m pytest tests/ -q

# 消融实验：纯向量 / 纯 BM25 / 混合检索 三路对比
python scripts/ablation.py
```

- 单元测试 27 个用例：意图路由、jieba 中文 BM25 检索、订单/物流/退款工具、投诉人工接管（interrupt/resume）、跨轮会话记忆、评测集 schema
- 消融实验（100 条评测集、TOP_K=4）：纯向量 96.0% / 纯 BM25 93.0% / **混合检索 98.0%**，混合路零回退（无纯向量命中而混合漏掉的条目）

## 进阶优化方向（写进简历的加分项）

1. **重排**：`.env` 开 `RERANK=1` 接入 bge-reranker，观察 top-4 命中率变化
2. **多轮记忆持久化**：把 `MemorySaver` 换成 SqliteSaver，支持跨会话重启
3. **成本优化**：结合 `UsageCallbackHandler` 数据，做命中缓存 / 更小模型 / 早停路由
