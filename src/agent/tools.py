"""业务工具（LangGraph 阶段注册给 LLM）。

当前为模拟订单数据，后续替换为真实订单服务即可（保持工具签名不变）。
每个 @tool 的 docstring 会作为函数 schema 描述交给模型，务必写清楚。
"""
from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

# 模拟订单表
MOCK_ORDERS = [
    {
        "order_id": "SO20260808001",
        "user_id": 1001,
        "product_name": "星火 AirPro 主动降噪蓝牙耳机",
        "status": "已发货",
        "logistics": "顺丰 SF1234567890",
        "amount": 499.0,
        "created_at": "2026-08-08",
    },
    {
        "order_id": "SO20260812002",
        "user_id": 1001,
        "product_name": "星火 27英寸 2K 显示器 M27",
        "status": "待发货",
        "logistics": None,
        "amount": 1299.0,
        "created_at": "2026-08-12",
    },
]


def _find_order(order_id: str) -> dict:
    for o in MOCK_ORDERS:
        if o["order_id"] == order_id:
            return o
    return {"error": f"未找到订单 {order_id}，请核对订单号"}


@tool
def query_order(order_id: str) -> str:
    """根据订单号查询订单状态（已发货/待发货/已取消）、商品名、金额与下单时间。"""
    o = _find_order(order_id)
    if "error" in o:
        return o["error"]
    return (
        f"订单 {o['order_id']}：{o['product_name']}，状态「{o['status']}」，"
        f"金额 {o['amount']} 元，下单时间 {o['created_at']}"
    )


@tool
def track_logistics(order_id: str) -> str:
    """根据订单号查询物流单号与当前物流节点。"""
    o = _find_order(order_id)
    if "error" in o:
        return o["error"]
    if not o.get("logistics"):
        return f"订单 {order_id} 尚未发货，暂无物流信息"
    return f"订单 {order_id} 物流单号 {o['logistics']}，当前节点：运输中，预计 2 天内送达"


@tool
def check_refund_eligibility(order_id: str) -> str:
    """校验订单是否满足直接退款条件。已发货订单需先退货流程，不可直接退款。"""
    o = _find_order(order_id)
    if "error" in o:
        return o["error"]
    eligible = o["status"] == "待发货"
    reason = "可直接退款" if eligible else "已发货订单需确认收货后再申请售后"
    return f"订单 {order_id} 退款资格：{'符合' if eligible else '不符合'}（{reason}）"


@tool
def apply_refund(order_id: str, reason: str) -> str:
    """为用户提交退款申请，返回退款金额与预计到账时间。"""
    o = _find_order(order_id)
    if "error" in o:
        return o["error"]
    return (
        f"退款申请已提交：订单 {order_id}，退款金额 {o['amount']} 元，"
        f"原因「{reason}」，预计 1-3 个工作日到账"
    )


@tool
def search_knowledge_base(query: str, state: Annotated[dict, InjectedState], k: int = 3) -> str:
    """检索「星火数码」的商品与售后知识库，返回相关商品参数、售后规则或 FAQ 原文。
    回答商品咨询、参数对比、售后政策、运费、退款时效等问题前，应优先调用本工具获取事实依据。"""
    from src.rag.retriever import get_retriever
    from src.rag.rewrite import resolve_query

    # 多轮时把"那款呢"这类指代改写成语义完整的问句再检索，否则第二句搜不到商品名
    resolved = resolve_query(state, query)
    docs = get_retriever().invoke(resolved)[:k]
    if not docs:
        return "知识库中未检索到相关信息。"
    parts = []
    for d in docs:
        name = d.metadata.get("name") or d.metadata.get("title") or d.metadata.get("source", "未知来源")
        parts.append(f"【{name}】\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


# 售前/售后 Agent 使用的工具集
ORDER_TOOLS = [query_order, track_logistics, check_refund_eligibility, apply_refund]
AGENT_TOOLS = ORDER_TOOLS + [search_knowledge_base]
