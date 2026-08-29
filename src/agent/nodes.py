"""LangGraph 各节点：意图路由、售前、售后、投诉升级（人工接管）、兜底。"""
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from src.agent.prompts import CHITCHAT_REPLY, ROUTER_SYSTEM, UPGRADE_SYSTEM
from src.rag.rewrite import build_history_text, needs_context, rewrite_query

INTENTS = ("presale", "aftersale", "complaint", "chitchat")


def classify_intent(user_input: str, llm) -> str:
    """判断用户输入意图，供路由节点与意图评测复用。"""
    try:
        resp = llm.invoke(ROUTER_SYSTEM.format(user_input=user_input))
        text = (getattr(resp, "content", None) or "").strip().lower()
        for it in INTENTS:
            if it in text:
                return it
    except Exception:
        pass
    return "chitchat"


def make_router_node(llm):
    """意图路由节点：只更新 state['intent']，不产生对话消息。"""

    def router_node(state):
        last = state["messages"][-1]
        text = last.content if isinstance(last.content, str) else str(last.content)
        # 短句/指代类追问（如"那它呢"）依赖上一轮上下文：直接继承上一轮意图，
        # 并把指代改写成语义完整的问句交给 Agent，避免被当成闲聊或答错对象
        history = build_history_text(state["messages"][:-1])
        if needs_context(text, history):
            prev = state.get("intent")
            if prev in INTENTS:
                updates = {"intent": prev, "resolved_query": ""}
                if prev in ("presale", "aftersale"):  # 只有 Agent 路径需要改写
                    updates["resolved_query"] = rewrite_query(llm, history, text)
                return updates
        return {"intent": classify_intent(text, llm), "resolved_query": ""}

    return router_node


def make_agent_node(agent):
    """通用节点：把 state 中的对话历史交给内嵌的 prebuilt agent，回传新消息。

    若路由已把指代改写为完整问句（state.resolved_query），用改写结果替换用户最后一句，
    让 Agent 明确知道在问什么，而不是面对"那它呢"含糊作答。
    """

    def node(state):
        messages = state["messages"]
        resolved = state.get("resolved_query")
        if resolved:
            last = messages[-1]
            original = last.content if isinstance(last.content, str) else str(last.content)
            replaced = HumanMessage(
                content=f"{resolved}（用户原话：{original}）", id=last.id
            )
            messages = messages[:-1] + [replaced]
        result = agent.invoke({"messages": messages})
        return {"messages": result["messages"]}

    return node


def make_upgrade_node(llm):
    """投诉升级节点：先安抚，再 interrupt 挂起等待人工客服处理结果。"""

    def upgrade_node(state):
        last = state["messages"][-1]
        user_text = last.content if isinstance(last.content, str) else str(last.content)
        ack = llm.invoke(UPGRADE_SYSTEM + "\n\n用户输入：" + user_text)
        ack_text = (getattr(ack, "content", None) or "很抱歉给您带来困扰，我先把您的问题转给人工客服处理。").strip()

        # 挂起图执行，把控制权交给人工侧；resume 后 interrupt() 返回人工处理结果
        human_decision = interrupt({
            "type": "human_handoff",
            "user_message": user_text,
            "pending_reply": ack_text,
        })

        if human_decision:
            final = f"{ack_text}\n\n【人工客服】{human_decision}"
        else:
            final = ack_text
        return {"messages": [AIMessage(content=final)]}

    return upgrade_node


def make_chitchat_node():
    """兜底节点：不调 LLM，固定话术。"""

    def chitchat_node(state):
        return {"messages": [AIMessage(content=CHITCHAT_REPLY)]}

    return chitchat_node
