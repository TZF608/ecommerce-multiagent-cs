"""测试用的假 LLM：不调真实 API，按 prompt 特征返回预设回复。

用来在离线环境跑状态机/路由测试，模拟 DeepSeek 等厂商返回。
"""
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeChatModel(BaseChatModel):
    """按 prompt 内容返回预设回复的假 LLM。

    - 路由提示（ROUTER_SYSTEM）：只看「用户输入：」之后的文本，按关键词返回意图类别；
    - 投诉升级提示（UPGRADE_SYSTEM）：返回固定安抚话术；
    - 其余调用：返回固定「好的」。
    """

    ack_text: str = "很抱歉给您带来困扰，我已将您的问题转给人工客服处理，会尽快跟进。"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        last = messages[-1]
        text = last.content if isinstance(last.content, str) else str(last.content)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self._reply_for(text)))])

    def _reply_for(self, text: str) -> str:
        if "质检员" in text:  # UPGRADE_SYSTEM：安抚话术
            return self.ack_text
        if "对话改写" in text:  # REWRITE_SYSTEM：把历史里提到的商品补进当前问句
            if "AirPro" in text:
                return "AirPro 和 AirMini 哪个续航久"
            if "机械键盘" in text:
                return "H75 机械键盘手感怎么样"
            return "请介绍下刚才那款商品"
        if "意图路由" in text:  # ROUTER_SYSTEM：只看用户输入，避免被系统提示里的示例词干扰
            idx = text.rfind("用户输入：")
            user = text[idx + len("用户输入："):] if idx != -1 else text
            if "投诉" in user:
                return "complaint"
            if "订单" in user or "退款" in user:
                return "aftersale"
            if "耳机" in user or "键盘" in user:
                return "presale"
            return "chitchat"
        return "好的"

    def bind_tools(self, tools, **kwargs):
        """create_react_agent 会调用 bind_tools；假模型直接返回自身。"""
        return self

    def bind(self, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "fake-chat"


class RaisingChatModel(FakeChatModel):
    """永远抛异常的假 LLM，用于验证 classify_intent 的兜底逻辑。"""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise RuntimeError("mock LLM failure")
