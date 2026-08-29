"""FastAPI 后端：多智能体客服的 HTTP 接口。

接口：
  POST /chat          {"message", "session_id?"} -> 客服回答（含成本/耗时/来源）
  POST /handoff/resume {"session_id", "human_reply"} -> 人工接管后的最终回复
  GET  /health
  GET  /             聊天前端页面

启动：uvicorn src.api.main:app --reload --port 8000
"""
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel

from src import config
from src.agent.callbacks import UsageCallbackHandler
from src.agent.graph import build_agent, get_interrupt

app = FastAPI(title="星火数码客服 Agent API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

_graph = None
_sessions: dict[str, str] = {}  # session_id -> thread_id


def get_graph():
    """懒加载：首次请求才构建 Agent（需要 .env 已配置 LLM）。"""
    global _graph
    if _graph is None:
        _graph = build_agent()
    return _graph


def _extract_sources(messages: list) -> list[str]:
    """从工具返回中提取命中的知识库来源名（用于给用户展示检索依据）。"""
    names = []
    for m in messages:
        if getattr(m, "type", "") == "tool":
            for line in (m.content or "").splitlines():
                if line.startswith("【") and "】" in line:
                    names.append(line[1: line.index("】")])
    return list(dict.fromkeys(names))


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class HandoffRequest(BaseModel):
    session_id: str
    human_reply: str


@app.get("/", include_in_schema=False)
def index():
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message 不能为空")

    session_id = req.session_id or uuid.uuid4().hex
    thread_id = _sessions.setdefault(session_id, uuid.uuid4().hex)
    graph = get_graph()

    cb = UsageCallbackHandler()
    cfg = {"configurable": {"thread_id": thread_id}, "callbacks": [cb]}
    t0 = time.perf_counter()
    try:
        result = graph.invoke({"messages": [HumanMessage(content=message)]}, config=cfg)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Agent 初始化失败，请检查 .env 的 LLM 配置：{e}")
    latency_ms = int((time.perf_counter() - t0) * 1000)

    interrupt = get_interrupt(result)
    if interrupt is not None:
        # 触发人工接管：返回待处理提示，前端可引导人工输入后调用 /handoff/resume
        payload = interrupt.value or {}
        return {
            "session_id": session_id,
            "reply": payload.get("pending_reply", "已为您转接人工客服。"),
            "intent": "complaint",
            "sources": [],
            "handoff_pending": True,
            "cost_usd": round(cb.cost_usd(config.LLM_MODEL), 6),
            "latency_ms": latency_ms,
            "tokens": {"prompt": cb.prompt_tokens, "completion": cb.completion_tokens},
        }

    last = result["messages"][-1]
    reply = last.content if isinstance(last.content, str) else str(last.content)
    return {
        "session_id": session_id,
        "reply": reply,
        "intent": result.get("intent"),
        "sources": _extract_sources(result["messages"]),
        "handoff_pending": False,
        "cost_usd": round(cb.cost_usd(config.LLM_MODEL), 6),
        "latency_ms": latency_ms,
        "tokens": {"prompt": cb.prompt_tokens, "completion": cb.completion_tokens},
    }


@app.post("/handoff/resume")
def handoff_resume(req: HandoffRequest):
    """人工客服处理结果写回对话（resume 被 interrupt 挂起的图）。"""
    thread_id = _sessions.get(req.session_id)
    if not thread_id:
        raise HTTPException(status_code=404, detail="session 不存在或已结束")
    graph = get_graph()
    cb = UsageCallbackHandler()
    cfg = {"configurable": {"thread_id": thread_id}, "callbacks": [cb]}
    t0 = time.perf_counter()
    human_reply = req.human_reply.strip()
    resume_val = human_reply or "（未收到人工处理结果，如需帮助请重新提问）"
    try:
        result = graph.invoke(Command(resume=resume_val), config=cfg)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    latency_ms = int((time.perf_counter() - t0) * 1000)
    last = result["messages"][-1]
    reply = last.content if isinstance(last.content, str) else str(last.content)
    return {
        "session_id": req.session_id,
        "reply": reply,
        "handoff_pending": False,
        "cost_usd": round(cb.cost_usd(config.LLM_MODEL), 6),
        "latency_ms": latency_ms,
    }
