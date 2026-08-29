"""检索效果评估（第 1 周即可跑，无需 LLM）。

统计两个指标：
- 关键词全部命中的问题占比（query 级 recall）
- 关键词平均命中率

用法：python scripts/evaluate.py [-k 4]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.rag.retriever import get_retriever  # noqa: E402


def load_queries() -> list[dict]:
    path = config.EVAL_DIR / "eval_queries.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="检索效果评估")
    parser.add_argument("-k", "--top-k", type=int, default=None, help="检索条数")
    args = parser.parse_args()
    top_k = args.top_k or config.TOP_K

    queries = load_queries()
    retriever = get_retriever()

    hit_all = 0
    total_keywords = 0
    hit_keywords = 0
    for item in queries:
        docs = retriever.invoke(item["query"])
        context = "\n".join(d.page_content for d in docs)
        keywords = item.get("expected_keywords", [])
        hits = [kw for kw in keywords if kw.lower() in context.lower()]
        total_keywords += len(keywords)
        hit_keywords += len(hits)
        ok = len(keywords) > 0 and len(hits) == len(keywords)
        hit_all += int(ok)
        flag = "OK  " if ok else "MISS"
        print(f"{flag} [{item['intent']:>16}] {item['query']}   命中 {len(hits)}/{len(keywords)}")

    n = len(queries)
    print("\n===== 检索评估结果 =====")
    print(f"评估问题数：{n}，TOP_K={top_k}")
    print(f"关键词全部命中的问题占比（query 级 recall）：{hit_all}/{n} = {hit_all / n:.1%}")
    print(f"关键词平均命中率：{hit_keywords}/{total_keywords} = {hit_keywords / total_keywords:.1%}")
    print("优化方向：调整 chunk 策略 / 开启混合检索 / 增加同义词，观察指标变化。")


if __name__ == "__main__":
    main()
