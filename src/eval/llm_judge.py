"""LLM-as-judge 答案评测：给智能客服回答打分（1-5），支持可量化的质量指标。"""
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.llm import get_llm

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是电商客服质检专家。给智能客服的回答打 1-5 分，只输出一个数字：\n"
        "5 完全正确、信息完整；4 正确，少量细节缺失；3 基本正确但有偏差；"
        "2 有错误信息；1 答非所问或编造。\n"
        "评分依据：与标准答案核对信息准确性，兼顾语气是否礼貌、能否解决问题。",
    ),
    ("human", "用户问题：{question}\n标准答案：{reference}\n客服回答：{answer}\n\n评分："),
])


def judge_answer(question: str, reference: str, answer: str) -> dict:
    """返回 {'score': int, 'judge_text': str}。"""
    llm = get_llm()
    chain = JUDGE_PROMPT | llm | StrOutputParser()
    text = chain.invoke({"question": question, "reference": reference, "answer": answer})
    match = re.search(r"[1-5]", text or "")
    score = int(match.group(0)) if match else 3
    return {"score": score, "judge_text": text.strip()}


def summarize(results: list[dict]) -> dict:
    """汇总评估结果：平均分、通过率（>=4 分）、以及平均成本与耗时。"""
    n = len(results)
    if not n:
        return {}
    avg_score = sum(r["score"] for r in results) / n
    pass_rate = sum(1 for r in results if r["score"] >= 4) / n
    avg_latency = sum(r["latency_ms"] for r in results) / n
    avg_tokens = sum(r["prompt_tokens"] + r["completion_tokens"] for r in results) / n
    total_cost = sum(r["cost"] for r in results)
    return {
        "评估条数": n,
        "平均分": round(avg_score, 2),
        "通过率(≥4分)": f"{pass_rate:.1%}",
        "平均耗时(ms)": round(avg_latency, 1),
        "平均token": round(avg_tokens, 1),
        "总成本(USD)": round(total_cost, 4),
        "单条平均成本(USD)": round(total_cost / n, 6),
    }
