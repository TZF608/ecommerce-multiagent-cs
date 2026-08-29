"""多轮检索 query 改写测试：指代补全逻辑与异常回退。"""
from langchain_core.messages import AIMessage, HumanMessage

from src.rag.rewrite import (
    build_history_text,
    needs_context,
    needs_rewrite,
    resolve_query,
    rewrite_query,
)


def test_needs_rewrite_requires_history():
    """没有历史（单轮）不改写，保证 100 条单轮评测不受影响。"""
    assert not needs_rewrite("那款呢", "")
    assert not needs_rewrite("AirPro耳机有降噪吗", "有历史")


def test_needs_rewrite_short_or_referential():
    assert needs_rewrite("那款呢", "有历史")          # 短
    assert needs_rewrite("这个多少钱", "有历史")       # 指代
    assert needs_rewrite("那它的降噪怎么样", "有历史")  # 指代 + 完整句


def test_build_history_text_keeps_recent_rounds():
    msgs = [
        HumanMessage("AirPro和AirMini哪个好"),
        AIMessage("推荐 AirPro，续航 40 小时"),
        HumanMessage("那续航呢"),
    ]
    text = build_history_text(msgs)
    assert "AirPro和AirMini" in text
    assert "推荐 AirPro" in text
    assert "那续航呢" in text


def test_rewrite_query_returns_resolved(fake_llm):
    out = rewrite_query(fake_llm, "用户: AirPro和AirMini哪个好", "那续航呢")
    assert "AirPro" in out
    assert "AirMini" in out


def test_rewrite_query_falls_back_on_error(raising_llm):
    """LLM 报错时回退原 query，不让检索中断。"""
    assert rewrite_query(raising_llm, "历史", "那款呢") == "那款呢"


def test_resolve_query_uses_history(fake_llm):
    state = {"messages": [
        HumanMessage("AirPro和AirMini哪个好"),
        AIMessage("推荐 AirPro"),
        HumanMessage("那续航呢"),
    ]}
    resolved = resolve_query(state, "那续航呢", llm=fake_llm)
    assert "AirPro" in resolved


def test_resolve_query_single_turn_untouched(fake_llm):
    state = {"messages": [HumanMessage("AirPro耳机有主动降噪吗")]}
    assert resolve_query(state, "AirPro耳机有主动降噪吗", llm=fake_llm) == "AirPro耳机有主动降噪吗"


def test_needs_context_is_conservative():
    """路由侧判断要保守：短句/指代开头才算追问，完整新话题不算。"""
    assert needs_context("那它呢", "有历史")
    assert needs_context("那续航呢", "有历史")
    assert needs_context("这个多少钱", "有历史")
    assert not needs_context("今天天气怎么样", "有历史")
    assert not needs_context("AirPro耳机有降噪吗", "有历史")
    assert not needs_context("那它呢", "")  # 无历史（首轮）不继承
