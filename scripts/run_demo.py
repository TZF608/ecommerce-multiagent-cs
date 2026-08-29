"""命令行 RAG 问答 demo。

用法：
  python scripts/run_demo.py                  交互式问答
  python scripts/run_demo.py -q "问题"        单次问答
  python scripts/run_demo.py --retrieve-only  只展示检索结果，不调用 LLM（未配置 LLM 也能用）
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.chain import build_chain  # noqa: E402
from src.rag.retriever import get_retriever  # noqa: E402


def print_sources(docs) -> None:
    print("\n===== 检索到的参考知识 =====")
    for i, doc in enumerate(docs, 1):
        name = doc.metadata.get("name") or doc.metadata.get("source", "未知来源")
        score = doc.metadata.get("retrieval_score", "-")
        preview = doc.page_content[:90].replace("\n", " ")
        print(f"[{i}] {name} | score={score}")
        print(f"    {preview}...")
    print("=" * 40)


def ask(question: str, retriever, chain, show_sources: bool = True) -> None:
    print(f"\n【用户】{question}")
    docs = retriever.invoke(question)
    if show_sources:
        print_sources(docs)
    if chain is not None:
        answer = chain.invoke(question)
        print(f"【客服】{answer}")


def main() -> None:
    parser = argparse.ArgumentParser(description="电商客服 RAG 问答 demo")
    parser.add_argument("-q", "--query", help="单次提问；不填则进入交互模式")
    parser.add_argument("--retrieve-only", action="store_true", help="只展示检索结果，不调用 LLM")
    parser.add_argument("-k", "--top-k", type=int, help="检索条数（覆盖 .env 的 TOP_K）")
    args = parser.parse_args()

    retriever = get_retriever()
    chain = None
    if not args.retrieve_only:
        # 未配置 LLM 时这里会抛错，报错信息里有 .env 配置指引
        chain = build_chain(retriever=retriever)

    if args.query:
        ask(args.query, retriever, chain)
        return

    print("进入交互模式，输入问题后回车（输入 q 退出）。")
    try:
        while True:
            q = input("\n问题> ").strip()
            if q.lower() in {"q", "quit", "exit"}:
                break
            if q:
                ask(q, retriever, chain)
    except (KeyboardInterrupt, EOFError):
        print("\n再见！")


if __name__ == "__main__":
    main()
