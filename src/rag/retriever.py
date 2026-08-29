"""检索器：向量检索 + 中文 BM25 混合（RRF 融合）+ 可选 CrossEncoder 重排。

中文关键词召回的关键点：langchain 自带的 BM25Retriever 按空格分词，
对中文会把整句当成一个 token、召回完全失效，所以这里用 jieba 分词后
再跑 rank_bm25。
"""
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from rank_bm25 import BM25Okapi

from src import config
from src.llm import get_embeddings


def _doc_key(doc: Document) -> str:
    """给文档一个稳定标识，用于跨路召回结果去重。"""
    return doc.metadata.get("product_id", "") + "|" + doc.page_content[:80]


def rrf(ranked_lists, k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion：多路召回结果按排名加权求和，取综合得分前 N。"""
    fused: dict[str, float] = {}
    for docs in ranked_lists:
        for rank, doc in enumerate(docs):
            key = _doc_key(doc)
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def tokenize_zh(text: str) -> list[str]:
    """中文分词（jieba），供 BM25 使用。"""
    import jieba

    return [t for t in jieba.lcut(text) if t.strip()]


class ChineseBM25:
    """基于 jieba 分词的 BM25 关键词检索。"""

    def __init__(self, docs: list[Document]):
        self.docs = docs
        self._tokenized = [tokenize_zh(d.page_content) for d in docs]
        self._bm25 = BM25Okapi(self._tokenized)

    def top_k(self, query: str, k: int) -> list[Document]:
        """返回 BM25 得分 > 0 的前 k 个文档；无命中时返回空列表（融合会退化为纯向量）。"""
        scores = self._bm25.get_scores(tokenize_zh(query))
        ranked = sorted(zip(self.docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, s in ranked if s > 0][:k]


class VectorRetriever:
    """基于 Chroma 的向量检索。"""

    def __init__(self):
        self.vectorstore = Chroma(
            persist_directory=str(config.CHROMA_DIR),
            embedding_function=get_embeddings(),
            collection_name="ecommerce_kb",
        )

    def retrieve(self, query: str, k: Optional[int] = None) -> list[Document]:
        k = k or config.TOP_K
        hits = self.vectorstore.similarity_search_with_score(query, k=k)
        docs = []
        for doc, score in hits:
            # 向量检索的距离，越小越相似
            doc.metadata["retrieval_score"] = round(float(score), 4)
            docs.append(doc)
        return docs


class HybridRetriever:
    """jieba 中文 BM25（关键词召回）+ 向量检索（语义召回），RRF 融合。

    解决两类典型问题：向量检索对专业术语/编号不敏感，BM25 对同义改写无计可施；
    两者互补后召回更稳。
    """

    def __init__(self):
        from src.data.loader import build_documents

        self.docs = build_documents()
        self._key_to_doc = {_doc_key(d): d for d in self.docs}
        self.bm25 = ChineseBM25(self.docs)
        self.vectorstore = Chroma(
            persist_directory=str(config.CHROMA_DIR),
            embedding_function=get_embeddings(),
            collection_name="ecommerce_kb",
        )

    def retrieve(self, query: str, k: Optional[int] = None) -> list[Document]:
        k = k or config.TOP_K
        bm25_hits = self.bm25.top_k(query, k * 2)
        vec_hits = self.vectorstore.similarity_search(query, k=k * 2)
        fused = rrf([bm25_hits, vec_hits])
        docs = []
        for key, score in fused[:k]:
            doc = self._key_to_doc[key]
            # 融合后的 RRF 得分，越高越相关
            doc.metadata["retrieval_score"] = round(float(score), 4)
            docs.append(doc)
        return docs


class Reranker:
    """CrossEncoder 重排（可选，需额外下载 bge-reranker 模型，体积较大）。"""

    def __init__(self, model: str = "BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model)

    def rerank(self, query: str, docs: list[Document], k: Optional[int] = None) -> list[Document]:
        k = k or config.TOP_K
        pairs = [[query, d.page_content] for d in docs]
        scores = self.model.predict(pairs)
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in scored[:k]]


def get_retriever():
    """按配置返回检索器（结果缓存，避免重复加载向量库），包装为 Runnable 以便接入 LCEL。"""
    global _retriever_cache
    if _retriever_cache is not None:
        return _retriever_cache

    retriever: object
    if config.USE_HYBRID:
        try:
            retriever = HybridRetriever()
        except ImportError:
            print("[警告] 未安装 rank-bm25 / jieba，回退到纯向量检索")
            retriever = VectorRetriever()
    else:
        retriever = VectorRetriever()
    _retriever_cache = RunnableLambda(retriever.retrieve)
    return _retriever_cache


_retriever_cache = None
