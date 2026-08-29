"""全局配置：从项目根目录 .env 读取，未设置时使用默认值。"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

# 国内直连 huggingface.co 不通：默认走 hf-mirror 镜像（已手动设置则尊重原配置）。
# 必须在 huggingface_hub 被 import 之前设置，故放最早加载的 config 模块。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ---- 目录 ----
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
EVAL_DIR = DATA_DIR / "eval"
CHROMA_DIR = DATA_DIR / "chroma"

# ---- LLM（OpenAI 兼容接口，可指向任意厂商）----
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# ---- Embedding ----
# huggingface：本地小模型（默认）；openai：走 LLM 同款 API 的嵌入接口
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
EMBEDDING_OPENAI_MODEL = os.getenv("EMBEDDING_OPENAI_MODEL", "text-embedding-3-small")

# ---- 检索参数 ----
TOP_K = int(os.getenv("TOP_K", "4"))
USE_HYBRID = os.getenv("USE_HYBRID", "1") == "1"
RERANK = os.getenv("RERANK", "0") == "1"
