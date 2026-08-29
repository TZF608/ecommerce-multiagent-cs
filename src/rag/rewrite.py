"""多轮检索的 query 改写：把"刚才那款耳机""那续航呢"这类指代补全成可独立检索的完整问句。

问题：检索只针对当前这一句 query。用户第二句说"那续航呢"，里面没有商品名，
检索拿不到任何东西——对话记忆能接住上下文，但搜索本身是断开的。
做法：检索前根据最近几轮对话决定是否改写，无需改写 / 改写失败时原样返回。
"""
from src.llm import get_llm

REWRITE_SYSTEM = """你是对话改写助手。多轮对话里用户可能用"这个""那款""它"等指代，请结合对话历史把当前问题改写成一条可独立检索的完整问句。
只输出改写后的问句，不要解释、不要加引号。当前问题本身已经完整时，原样输出。

对话历史：
{history}

当前问题：{query}
改写后："""

# 出现这些词说明 query 可能依赖上下文，值得走改写（检索侧可激进：误判只多一次改写调用）
REFERENTIAL_HINTS = ("这个", "那个", "这款", "那款", "它", "那", "这", "呢", "咋", "啥", "多少", "怎么")
# query 短于该长度视为"不完整问句"，即使没有明显指代词也改写
SHORT_QUERY_LIMIT = 5

# 路由侧用的保守词表：只有短句或以指代词开头的问句才视为依赖上下文。
# 检索侧误判无害（改写后原样返回），但路由侧误判会把"今天天气怎么样"继承成上一轮意图，所以必须保守。
ROUTER_CONTEXT_HINTS = ("这个", "那个", "这款", "那款", "它", "那", "这")
ROUTER_SHORT_LIMIT = 6


def needs_context(query: str, history: str) -> bool:
    """路由用：当前问句是否依赖上一轮上下文（短句 / 指代词开头）。"""
    if not history:
        return False
    q = query.strip()
    if not q:
        return False
    if len(q) <= ROUTER_SHORT_LIMIT:
        return True
    return any(q.startswith(w) for w in ROUTER_CONTEXT_HINTS)


def build_history_text(messages: list, max_rounds: int = 2) -> str:
    """把最近几轮对话压成一段文本，供改写时参考。"""
    lines = []
    for msg in messages[-max_rounds * 2:]:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        role = "用户" if msg.type == "human" else ("客服" if msg.type == "ai" else msg.type)
        # 只取前 80 字，防止工具返回的长文本灌进改写 prompt
        lines.append(f"{role}: {content[:80]}")
    return "\n".join(lines)


def needs_rewrite(query: str, history: str) -> bool:
    """有历史，且当前问题短或含指代词 → 需要改写。"""
    if not history:
        return False
    q = query.strip()
    if not q:
        return False
    if len(q) <= SHORT_QUERY_LIMIT:
        return True
    return any(hint in q for hint in REFERENTIAL_HINTS)


def rewrite_query(llm, history: str, query: str) -> str:
    """用 LLM 把 query 结合历史改写成独立问句；任何异常都回退原 query。"""
    try:
        resp = llm.invoke(REWRITE_SYSTEM.format(history=history, query=query))
        text = (getattr(resp, "content", None) or "").strip()
        if text and len(text) <= 200:  # 防御：改写结果过长视为异常
            return text
    except Exception:
        pass
    return query


def resolve_query(state: dict, query: str, llm=None) -> str:
    """检索入口：根据对话历史决定是否改写 query。返回最终用于检索的问句。"""
    history = build_history_text(state.get("messages") or [])
    if needs_rewrite(query, history):
        return rewrite_query(llm or get_llm(), history, query)
    return query
