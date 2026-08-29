"""Token 用量统计与成本估算（用于评估单会话成本、简历量化数据）。"""
from langchain_core.callbacks import BaseCallbackHandler

# 粗粒度价格表（美元 / 百万 token）：(模型前缀, 输入价, 输出价)。仅用于估算。
PRICE_TABLE = [
    ("gpt-4o", 2.5, 10.0),
    ("gpt-4.1", 2.0, 8.0),
    ("gpt-4", 30.0, 60.0),
    ("gpt-3.5", 0.5, 1.5),
    ("deepseek", 0.27, 1.10),
    ("glm", 0.5, 2.0),
    ("qwen", 0.3, 1.2),
    ("moonshot", 2.0, 6.0),
]


class UsageCallbackHandler(BaseCallbackHandler):
    """汇总一次对话中所有 LLM 调用的 token 用量。

    用法：把实例放进 graph.invoke / llm.invoke 的 config["callbacks"]。
    """

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage", {}) or {}
        self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)

    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def cost_usd(self, model: str = "") -> float:
        """按模型前缀估算美元成本；无用量返回 0。"""
        if not self.total_tokens():
            return 0.0
        in_price, out_price = 2.5, 10.0
        for prefix, pi, po in PRICE_TABLE:
            if prefix in model:
                in_price, out_price = pi, po
                break
        return (
            self.prompt_tokens / 1_000_000 * in_price
            + self.completion_tokens / 1_000_000 * out_price
        )
