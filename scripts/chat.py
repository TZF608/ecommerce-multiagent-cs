"""完整多智能体客服对话 CLI（需要配置 LLM）。

用法：python scripts/chat.py
特性：意图路由 / 售前 RAG / 售后工具调用 / 投诉自动转人工（可输入人工处理结果恢复）。
"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402

from src import config  # noqa: E402
from src.agent.callbacks import UsageCallbackHandler  # noqa: E402
from src.agent.graph import build_agent, get_interrupt  # noqa: E402


def main() -> None:
    graph = build_agent()
    thread_id = uuid.uuid4().hex
    print("星火数码智能客服（完整 Agent）。输入问题回车；输入 q 退出。")

    while True:
        try:
            q = input("\n你> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break
        if not q:
            continue
        if q.lower() in {"q", "quit", "exit"}:
            break

        cb = UsageCallbackHandler()
        cfg = {"configurable": {"thread_id": thread_id}, "callbacks": [cb]}
        result = graph.invoke({"messages": [HumanMessage(content=q)]}, config=cfg)

        interrupt = get_interrupt(result)
        if interrupt is not None:
            payload = interrupt.value or {}
            print(f"\n客服> {payload.get('pending_reply', '已为您转接人工客服。')}")
            print("      [已转接人工客服 · 演示] 输入人工处理结果后回车；直接回车则结束本次接管")
            try:
                human = input("人工> ").strip()
            except (KeyboardInterrupt, EOFError):
                human = ""
            resume_val = human or "（未收到人工处理结果，如需帮助请重新提问）"
            result = graph.invoke(Command(resume=resume_val), config=cfg)

        final = result["messages"][-1]
        reply = final.content if isinstance(final.content, str) else str(final.content)
        print(f"客服> {reply}")
        cost = cb.cost_usd(config.LLM_MODEL)
        print(f"      (本轮 tokens: {cb.prompt_tokens}+{cb.completion_tokens}, 估算成本 ${cost:.5f})")


if __name__ == "__main__":
    main()
