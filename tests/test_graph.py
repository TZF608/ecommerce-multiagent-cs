"""状态机端到端测试：意图路由、投诉人工接管（interrupt/resume）、跨轮记忆。

全部使用假 LLM，离线可跑。
"""
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.agent.graph import build_agent, get_interrupt


def _cfg(thread_id):
    return {"configurable": {"thread_id": thread_id}}


def test_complaint_interrupt_and_resume(fake_llm):
    graph = build_agent(fake_llm)
    result = graph.invoke(
        {"messages": [HumanMessage(content="我要投诉，耳机用了两周就坏了！")]},
        config=_cfg("t-complaint"),
    )
    # 投诉分支应挂起，而不是给出普通回答
    intr = get_interrupt(result)
    assert intr is not None
    assert intr.value["type"] == "human_handoff"
    assert "耳机" in intr.value["user_message"]

    # 人工结论通过 Command(resume=...) 合入对话
    result2 = graph.invoke(
        Command(resume="已登记，48小时内专员跟进。"), config=_cfg("t-complaint")
    )
    assert get_interrupt(result2) is None
    last = result2["messages"][-1].content
    assert "【人工客服】" in last
    assert "已登记，48小时内专员跟进。" in last


def test_chitchat_route(fake_llm):
    graph = build_agent(fake_llm)
    result = graph.invoke({"messages": [HumanMessage(content="你好呀")]}, config=_cfg("t-chat"))
    assert get_interrupt(result) is None
    assert "智能客服" in result["messages"][-1].content


def test_multiturn_memory(fake_llm):
    """同一 thread_id 下消息会累积，验证 MemorySaver 会话记忆。"""
    graph = build_agent(fake_llm)
    graph.invoke({"messages": [HumanMessage(content="你好")]}, config=_cfg("t-mem"))
    result = graph.invoke({"messages": [HumanMessage(content="谢谢")]}, config=_cfg("t-mem"))
    assert len(result["messages"]) == 4  # Human, AI, Human, AI


def test_get_interrupt_returns_none_on_normal_path(fake_llm):
    graph = build_agent(fake_llm)
    result = graph.invoke({"messages": [HumanMessage(content="你好呀")]}, config=_cfg("t-none"))
    assert get_interrupt(result) is None


def test_referential_followup_keeps_previous_intent(fake_llm):
    """"那它呢"这类短句追问继承上一轮售前意图，而不是被当成闲聊。"""
    graph = build_agent(fake_llm)
    cfg = _cfg("t-ref")
    graph.invoke({"messages": [HumanMessage(content="AirPro耳机有降噪吗")]}, config=cfg)
    result = graph.invoke({"messages": [HumanMessage(content="那它呢")]}, config=cfg)
    assert result["intent"] == "presale"
    assert "智能客服" not in result["messages"][-1].content  # 没走到闲聊兜底


def test_new_topic_not_swallowed_by_previous_intent(fake_llm):
    """完整新话题（"今天天气怎么样"）不该被误判成追问而继承上一轮意图。"""
    graph = build_agent(fake_llm)
    cfg = _cfg("t-new")
    graph.invoke({"messages": [HumanMessage(content="AirPro耳机有降噪吗")]}, config=cfg)
    result = graph.invoke({"messages": [HumanMessage(content="今天天气怎么样")]}, config=cfg)
    assert result["intent"] == "chitchat"
