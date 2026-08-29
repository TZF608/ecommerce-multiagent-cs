# 演示手册（Demo Runbook）

面试 / 评审 / 给别人展示「星火数码 · 电商智能客服 Agent」的完整流程。
按本手册走，5 分钟讲完全部亮点。**演示前先读「0. 前提」确认环境就绪。**

## 0. 前提（一次性准备，已就绪）

| 项 | 说明 |
|---|---|
| 环境 | `d:\project\.venv` 已建、依赖已装 |
| 配置 | `.env` 已填 DeepSeek key |
| 数据 | `data/chroma` 向量库、`data/eval/eval_queries.json`（100 条） |
| 国内网络 | `src/config.py` 已内置 hf-mirror 镜像，直接跑不卡 |

## 1. 启动 Web（2 分钟）

**方式 A（推荐）**：双击 `start_web.bat`，浏览器自动打开。

**方式 B**：命令行
```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --port 8000
```
浏览器打开 `http://127.0.0.1:8000`。

> ⚠️ 保持启动窗口开启，关闭即停止服务。
> ⚠️ 演示前先发一句话**预热**（首次加载模型约 2-3 秒）。

## 2. 五分钟演示讲稿（边说边点）

| 顺序 | 你输入 | 边点边说 |
|---|---|---|
| 1 | `AirPro 耳机支持降噪吗` | 这是**意图路由**，它判断为售前咨询 |
| 2 | `那它和 AirMini 哪个续航久` | **多轮记忆**——它记得上一轮在聊 AirPro |
| 3 | `我的订单 SO20260808001 到哪了` | **工具调用**——订单查询走了真实查询工具 |
| 4 | `我要投诉！` | **人工接管**——Agent 主动挂起交给真人，不是硬答 |
| 5 | 人工框填 `已登记换新` | **恢复合入对话**——human-in-the-loop 闭环 |

收尾强调成本：「刚才这 4 轮一共花了**不到 3 分钱**。」

## 3. CLI 演示（可选，技术面效果好）

```powershell
.\.venv\Scripts\python.exe scripts/chat.py
```
直接看到路由分支、token 用量、成本实时打印，适合讲架构。

## 4. 三条评测命令（证明数字，面试前跑一遍留底）

```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py      # 检索召回率（免费）
.\.venv\Scripts\python.exe scripts/eval_intents.py  # 意图准确率（需 key）
.\.venv\Scripts\python.exe scripts/eval_answers.py  # 端到端答案通过率（需 key，约 5-10 分钟）
```

实测数字（2026-08，DeepSeek，100 条评测集）：召回率 98% / 关键词命中 98.5% / 意图 98.9% / 答案通过率 100% / 平均分 4.94 / 单条 $0.00077。

消融实验（`python scripts/ablation.py`，可选补充）：纯向量 96% / 纯 BM25 93% / 混合 98%（零回退）。

## 5. 故障排查

| 现象 | 解法 |
|---|---|
| 端口被占用 | 换端口：`--port 8001`，浏览器开 `http://127.0.0.1:8001` |
| Web 打不开 | 服务没跑（窗口关了）→ 重新启动 |
| 模型加载慢 | 首次 2-3 秒正常，演示前预热 |
| 换电脑 | 重装依赖（README）+ 拷 `data/chroma` + 配 `.env`，无需重下模型 |

## 6. 远程演示

- 启动加 `--host 0.0.0.0`，把本机 IP 发给对方（同局域网）
- 或直接**录屏 3 分钟**发送
