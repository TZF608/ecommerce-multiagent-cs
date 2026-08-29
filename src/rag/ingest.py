"""把原始数据分块、向量化、写入 Chroma 向量库。"""
import shutil

from langchain_chroma import Chroma

from src import config
from src.data.loader import build_documents
from src.llm import get_embeddings


def ingest() -> Chroma:
    """全量重建向量库。每次重跑会先清空旧库，保证无重复数据。"""
    docs = build_documents()
    if not docs:
        raise RuntimeError("没有读到任何文档，请检查 data/raw 目录")

    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"加载文档块：{len(docs)} 个")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
        collection_name="ecommerce_kb",
    )
    print(f"向量库已写入：{config.CHROMA_DIR}")
    return vectorstore
