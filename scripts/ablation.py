"""消融实验：纯向量 / 纯 BM25 / 混合检索（jieba BM25 + 向量 + RRF）在评测集上的对比。

验证混合检索相对单一检索的提升幅度，结果写进简历。
用法：python scripts/ablation.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data.loader import build_documents  # noqa: E402
from src.rag.retriever import ChineseBM25, HybridRetriever, VectorRetriever  # noqa: E402


def load_queries() -> list[dict]:
    return json.load(open(config.EVAL_DIR / "eval_queries.json", encoding="utf-8"))


def evaluate(mode: str, queries: list[dict], k: int) -> tuple[int, int, int, list[str]]:
    """对评测集跑一遍检索，返回 (query 级命中数, 总关键词, 命中关键词, 差异示例)。"""
    if mode == "hybrid":
        retriever = HybridRetriever().retrieve
    elif mode == "vector":
        retriever = VectorRetriever().retrieve
    elif mode == "bm25":
        bm25 = ChineseBM25(build_documents())
        retriever = bm25.top_k
    else:
        raise ValueError(mode)

    hit_all = 0
    total_kw = 0
    hit_kw = 0
    diff_examples = []
    for item in queries:
        docs = retriever(item["query"], k)
        context = "\n".join(d.page_content for d in docs)
        keywords = item.get("expected_keywords", [])
        hits = [kw for kw in keywords if kw.lower() in context.lower()]
        total_kw += len(keywords)
        hit_kw += len(hits)
        ok = len(keywords) > 0 and len(hits) == len(keywords)
        hit_all += int(ok)
        # 记录与「默认混合」不一致的条目（在 main 里做差集）
        diff_examples.append((item["id"], item["query"], ok))
    return hit_all, total_kw, hit_kw, diff_examples


def main() -> None:
    queries = load_queries()
    k = config.TOP_K
    n = len(queries)

    print(f"评测集：{n} 条，TOP_K={k}\n")

    results = {}
    examples = {}
    for mode in ("vector", "bm25", "hybrid"):
        hit_all, total_kw, hit_kw, diff = evaluate(mode, queries, k)
        results[mode] = (hit_all, hit_kw, total_kw)
        examples[mode] = set(i for i, q, ok in diff if ok)
        print(f"[{mode:>6}] query 级 recall：{hit_all}/{n} = {hit_all / n:.1%}"
              f"    关键词命中率：{hit_kw}/{total_kw} = {hit_kw / total_kw:.1%}")

    hybrid_ok, vector_ok = examples["hybrid"], examples["vector"]
    print("\n仅混合检索命中、纯向量漏掉的条目：")
    for q in queries:
        if q["id"] in (hybrid_ok - vector_ok):
            print(f"  {q['id']}  {q['query']}  ->  {q['expected_keywords']}")
    print("仅纯向量命中、混合漏掉的条目：")
    for q in queries:
        if q["id"] in (vector_ok - hybrid_ok):
            print(f"  {q['id']}  {q['query']}  ->  {q['expected_keywords']}")


if __name__ == "__main__":
    main()
