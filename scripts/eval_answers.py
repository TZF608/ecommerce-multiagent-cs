"""端到端答案评测（需要配置 LLM，会消耗较多 token）。

流程：100 条评估集 -> 智能客服 Agent 回答 -> LLM-as-judge 打分 -> 汇总指标。
用法：python scripts/eval_answers.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402

from src import config  # noqa: E402
from src.agent.callbacks import UsageCallbackHandler  # noqa: E402
from src.agent.graph import build_agent, get_interrupt  # noqa: E402
from src.eval.llm_judge import judge_answer, summarize  # noqa: E402


def main() -> None:
    queries = json.load(open(config.EVAL_DIR / "eval_queries.json", encoding="utf-8"))
    graph = build_agent()

    results = []
    for q in queries:
        cb = UsageCallbackHandler()
        cfg = {"configurable": {"thread_id": f"eval-{q['id']}"}, "callbacks": [cb]}
        t0 = time.perf_counter()
        result = graph.invoke({"messages": [HumanMessage(content=q["query"])]}, config=cfg)
        # 投诉类会触发人工接管，模拟人工给出结论
        if get_interrupt(result) is not None:
            result = graph.invoke(Command(resume="已登记，人工客服 3 小时内跟进。"), config=cfg)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        answer = result["messages"][-1].content
        verdict = judge_answer(q["query"], q["reference_answer"], answer)
        cost = cb.cost_usd(config.LLM_MODEL)
        results.append({
            **q,
            "answer": answer,
            "score": verdict["score"],
            "latency_ms": latency_ms,
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "cost": cost,
        })
        flag = "OK " if verdict["score"] >= 4 else "LOW"
        print(f"{flag} [{q['id']}] 评分{verdict['score']}  {q['query']}")

    print("\n===== 答案评测汇总 =====")
    for k, v in summarize(results).items():
        print(f"{k}: {v}")

    # 输出明细，方便复查
    out = config.DATA_DIR / "eval" / "results_answers.json"
    out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n明细已写入：{out}")


if __name__ == "__main__":
    main()
