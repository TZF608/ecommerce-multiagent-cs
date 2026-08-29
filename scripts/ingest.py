"""构建向量库：分块 -> 向量化 -> 写入 Chroma。

用法：python scripts/ingest.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.ingest import ingest  # noqa: E402


if __name__ == "__main__":
    ingest()
