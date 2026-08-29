"""RAG 问答链：检索 -> 组装上下文 -> LLM 生成。"""
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from src.llm import get_llm
from src.rag.retriever import get_retriever

SYSTEM_PROMPT = """你是一家电商平台「星火数码」的智能客服助手。请严格依据「参考知识」回答用户问题。

要求：
1. 只依据参考知识回答，不编造知识库之外的信息；
2. 回答口语化、简洁、礼貌，像真人客服；
3. 若参考知识不足以回答，明确回复“这个问题我暂时无法确认，建议联系人工客服”；
4. 涉及价格、库存、时效、政策等数字时，严格以参考知识为准，不要自行推算。"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "参考知识：\n{context}\n\n---\n用户问题：{question}"),
])


def _format_context(docs) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


def build_chain(retriever: Optional[Runnable] = None) -> Runnable:
    """构建 RAG 链：retriever -> format -> prompt -> llm -> str。

    直接 `chain.invoke("用户问题")` 即可得到答案。
    """
    retriever = retriever or get_retriever()
    llm = get_llm()
    return (
        {"context": retriever | _format_context, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
