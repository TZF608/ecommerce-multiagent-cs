"""评测集 schema 校验：保证 data/eval/eval_queries.json 能被三条评测脚本消费。

- evaluate.py      读 query / expected_keywords / intent
- eval_intents.py  读 intent（映射到路由类别）
- eval_answers.py  读 id / query / reference_answer
"""
import json

from src import config
from src.agent.tools import MOCK_ORDERS
from src.data.loader import build_documents

from scripts.eval_intents import INTENT_TO_ROUTER  # noqa: E402

REQUIRED_FIELDS = {"id", "intent", "query", "expected_keywords", "reference_answer"}
KNOWN_INTENTS = set(INTENT_TO_ROUTER) | {"general_faq"}
# 工具域词：订单/物流状态，不在知识库文本里，但由订单工具返回
TOOL_DOMAIN_WORDS = {"运输中", "已发货", "待发货", "已取消", "顺丰", "预计"}


def load_queries():
    return json.load(open(config.EVAL_DIR / "eval_queries.json", encoding="utf-8"))


def test_has_100_queries():
    assert len(load_queries()) == 100


def test_required_fields_present():
    for q in load_queries():
        assert REQUIRED_FIELDS <= set(q), f"{q['id']} 缺字段"


def test_unique_ids():
    ids = [q["id"] for q in load_queries()]
    assert len(ids) == len(set(ids))


def test_intents_are_known():
    for q in load_queries():
        assert q["intent"] in KNOWN_INTENTS, f"{q['id']} 意图未知：{q['intent']}"


def test_query_non_empty():
    for q in load_queries():
        assert q["query"].strip(), q["id"]


def test_keywords_exist_in_corpus():
    """每个 expected_keyword 要么出现在知识库原文、要么在订单/物流工具域内。

    防止关键词凭空编造（既不在知识库、也不是工具能返回的词），
    否则该条会在 evaluate.py 里永远 MISS，拉低指标却没有意义。
    """
    kb_text = "\n".join(d.page_content for d in build_documents()).lower()
    order_text = json.dumps(MOCK_ORDERS, ensure_ascii=False).lower()
    for q in load_queries():
        for kw in q["expected_keywords"]:
            k = kw.lower()
            assert (k in kb_text) or (k in order_text) or (k in TOOL_DOMAIN_WORDS), \
                f"{q['id']} 关键词「{kw}」不在知识库/订单域内"
