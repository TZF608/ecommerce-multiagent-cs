"""统一的 LLM / Embedding 工厂。换厂商只改 .env，不改代码。"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from src import config


def get_llm(**kwargs) -> ChatOpenAI:
    """返回 ChatOpenAI，兼容 DeepSeek / Kimi / 通义 / 智谱 / Ollama / OpenAI。"""
    if not config.LLM_API_KEY and not config.LLM_BASE_URL:
        raise ValueError(
            "未配置 LLM：请把项目根目录的 .env.example 复制为 .env，并填写 LLM_API_KEY。\n"
            "支持 DeepSeek / Kimi / 通义 / 智谱 / OpenAI / Ollama，配置示例见 .env.example 注释。"
        )
    return ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=kwargs.get("temperature", config.LLM_TEMPERATURE),
        api_key=config.LLM_API_KEY or "EMPTY",
        base_url=config.LLM_BASE_URL or None,
    )


def get_embeddings():
    """返回 Embedding 模型。默认本地 huggingface，可切换为 API 嵌入。"""
    if config.EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.EMBEDDING_OPENAI_MODEL,
            api_key=config.LLM_API_KEY or "EMPTY",
            base_url=config.LLM_BASE_URL or None,
        )
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
