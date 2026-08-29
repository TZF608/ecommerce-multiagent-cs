"""LangGraph 多智能体工作流。

状态机：

    user_input
        │
        ▼
   [router 意图路由]  ──► chitchat ──► 兜底回复
        │
        ├─ presale   ──► [presale_agent]   （agentic RAG：search_knowledge_base 工具）
        ├─ aftersale ──► [aftersale_agent] （订单工具 + 知识库检索）
        └─ complaint ─► [upgrade_agent]    （interrupt 人工接管）

特性：
- 会话记忆：MemorySaver 按 thread_id 保存整段对话；
- 人工接管：complaint 分支挂起，resume 后由人工客服接管；
- 可注入 LLM：build_agent(llm=...) 便于测试与替换厂商。
"""
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

from src.agent.nodes import (
    make_agent_node,
    make_chitchat_node,
    make_router_node,
    make_upgrade_node,
)
from src.agent.prompts import AFTERSALE_SYSTEM, PRESALE_SYSTEM
from src.agent.tools import AGENT_TOOLS, ORDER_TOOLS, search_knowledge_base
from src.llm import get_llm


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: str
    # 多轮指代改写后的完整问句（如"那它呢"→"AirPro 的续航是多少"）；无改写为空字符串
    resolved_query: str


def get_interrupt(result):
    """从 graph.invoke 返回值中提取中断对象（langgraph 1.x 返回含 `__interrupt__` 键的 dict）。

    无中断时返回 None；有中断时返回 Interrupt 对象（.value 为 payload dict）。
    """
    if isinstance(result, dict) and result.get("__interrupt__"):
        return result["__interrupt__"][0]
    return None


def build_agent(llm=None):
    """构建并编译多智能体图。不传 llm 时按 .env 配置创建。"""
    llm = llm or get_llm()

    presale_agent = create_react_agent(
        llm,
        tools=[search_knowledge_base],
        prompt=PRESALE_SYSTEM,
        name="presale_agent",
    )
    aftersale_agent = create_react_agent(
        llm,
        tools=AGENT_TOOLS,  # 订单工具 + 知识库检索
        prompt=AFTERSALE_SYSTEM,
        name="aftersale_agent",
    )

    graph = StateGraph(AgentState)
    graph.add_node("router", make_router_node(llm))
    graph.add_node("presale", make_agent_node(presale_agent))
    graph.add_node("aftersale", make_agent_node(aftersale_agent))
    graph.add_node("upgrade", make_upgrade_node(llm))
    graph.add_node("chitchat", make_chitchat_node())

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        lambda state: state.get("intent", "chitchat"),
        {
            "presale": "presale",
            "aftersale": "aftersale",
            "complaint": "upgrade",
            "chitchat": "chitchat",
        },
    )
    for node in ("presale", "aftersale", "upgrade", "chitchat"):
        graph.add_edge(node, END)

    return graph.compile(checkpointer=MemorySaver())
