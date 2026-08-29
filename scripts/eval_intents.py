"""意图路由准确率评估（需要配置 LLM）。

用法：python scripts/eval_intents.py
说明：general_faq 类问题路由目标不唯一，单独列出预测、不计入准确率。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.agent.nodes import classify_intent  # noqa: E402
from src.llm import get_llm  # noqa: E402

# 细粒度意图 -> 路由类别
INTENT_TO_ROUTER = {
    "product_info": "presale",
    "product_compare": "presale",
    "availability_stock": "presale",
    "price": "presale",
    "shipping": "aftersale",
    "track_order": "aftersale",
    "return_policy": "aftersale",
    "refund": "aftersale",
    "warranty": "aftersale",
    "invoice": "aftersale",
    "complaint": "complaint",
    # "general_faq" 不映射：路由目标不唯一，单独统计
}


def main() -> None:
    queries = json.load(open(config.EVAL_DIR / "eval_queries.json", encoding="utf-8"))
    llm = get_llm()

    correct = total = 0
    ambiguous = []
    confusion: dict[str, int] = {}
    for q in queries:
        exp = INTENT_TO_ROUTER.get(q["intent"])
        pred = classify_intent(q["query"], llm)
        if exp is None:
            ambiguous.append((q["query"], pred))
            continue
        total += 1
        ok = pred == exp
        correct += int(ok)
        confusion[f"{exp}->{pred}"] = confusion.get(f"{exp}->{pred}", 0) + 1
        print(f"{'OK  ' if ok else 'MISS'} 期望={exp:9s} 预测={pred:9s}  {q['query']}")

    print("\n===== 意图路由评估 =====")
    print(f"计入统计：{total} 条（general_faq 排除）")
    print(f"意图准确率：{correct}/{total} = {correct / total:.1%}")
    print("混淆（期望->预测）：", {k: v for k, v in sorted(confusion.items(), key=lambda x: -x[1]) if v})
    if ambiguous:
        print("general_faq 预测（需人工判断）：")
        for query, pred in ambiguous:
            print(f"  {pred:9s}  {query}")


if __name__ == "__main__":
    main()
