"""检索器单元测试。

不加载 embedding 模型（避免下载/占用），用假文档验证
jieba 中文 BM25 关键词召回与 RRF 融合的去重排序逻辑。
"""
from langchain_core.documents import Document

from src.rag.retriever import ChineseBM25, _doc_key, rrf, tokenize_zh


def test_tokenize_zh_splits_chinese():
    tokens = tokenize_zh("AirPro耳机主动降噪")
    assert len(tokens) >= 3
    assert "AirPro" in tokens


def test_bm25_top_k_returns_keyword_doc():
    docs = [
        Document(page_content="AirPro耳机主动降噪蓝牙耳机 价格499元"),
        Document(page_content="移动电源20000mAh 超薄便携 支持22.5W快充"),
        Document(page_content="机械键盘H75 红轴 支持三模连接"),
    ]
    bm25 = ChineseBM25(docs)
    hits = bm25.top_k("AirPro耳机", k=2)
    assert len(hits) == 1
    assert hits[0] is docs[0]


def test_bm25_no_hit_returns_empty():
    """BM25 无命中时返回空列表（融合时会退化为纯向量），而非兜底所有文档。"""
    docs = [Document(page_content="移动电源20000mAh")]
    bm25 = ChineseBM25(docs)
    assert bm25.top_k("机械键盘", k=2) == []


def test_rrf_doc_in_both_lists_ranks_first():
    a = Document(page_content="AirPro耳机")
    b = Document(page_content="移动电源")
    c = Document(page_content="机械键盘")
    ranked = rrf([[a, b], [b, c]])
    # 出现在两路的 b 综合得分最高
    assert ranked[0][0] == _doc_key(b)
    assert ranked[0][1] > ranked[1][1]


def test_doc_key_distinguishes_same_content():
    """同一文案但不同商品应视为不同文档，融合不去重误伤。"""
    d1 = Document(page_content="耳机", metadata={"product_id": "p1"})
    d2 = Document(page_content="耳机", metadata={"product_id": "p2"})
    assert _doc_key(d1) != _doc_key(d2)
