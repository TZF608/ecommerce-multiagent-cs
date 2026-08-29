"""意图路由测试：classify_intent 的关键词路由与异常兜底。"""
import pytest

from src.agent.nodes import classify_intent


@pytest.mark.parametrize("query,expected", [
    ("AirPro耳机有主动降噪吗", "presale"),
    ("我的订单SO20260808001物流到哪了", "aftersale"),
    ("我要投诉！你们服务太差了", "complaint"),
    ("今天天气怎么样", "chitchat"),
])
def test_classify_intent_routing(fake_llm, query, expected):
    assert classify_intent(query, fake_llm) == expected


def test_classify_intent_fallback_on_error(raising_llm):
    """LLM 调用异常时兜底到 chitchat，不让整个对话崩溃。"""
    assert classify_intent("随便问问", raising_llm) == "chitchat"
