"""把 data/raw 下的原始数据转换成检索用的 Document 列表。"""
import json
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config


def load_products() -> list[Document]:
    """读取商品库 products.jsonl，一个商品 = 一个 Document。"""
    docs = []
    path = config.RAW_DIR / "products.jsonl"
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = "\n".join([
                f"商品名称：{item['name']}",
                f"分类：{item['category']}",
                f"价格：{item['price']} 元",
                f"库存：{item['stock']} 件",
                "规格：" + "，".join(f"{k} {v}" for k, v in item.get("specs", {}).items()),
                "卖点：" + "；".join(item.get("highlights", [])),
                f"详情：{item.get('description', '')}",
            ])
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": "products.jsonl",
                    "type": "product",
                    "product_id": item["id"],
                    "name": item["name"],
                    "category": item["category"],
                    "price": item["price"],
                },
            ))
    return docs


def load_markdown(filename: str, doc_type: str) -> list[Document]:
    """按 '## ' 小节切分 markdown 文档（售后规则 / FAQ），每节一个 Document。

    说明：规则/FAQ 的每个小节是一个独立主题，切碎后合并会稀释检索精度，
    因此这里每小节单独成块，并把小节标题存入 metadata 便于展示与溯源。
    """
    text = (config.RAW_DIR / filename).read_text(encoding="utf-8")
    parts = re.split(r"(?m)^(?=## )", text)

    docs = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(r"##\s+(.+)", part)
        title = m.group(1).strip() if m else filename
        if len(part) <= 800:
            chunks = [part]
        else:
            # 超长小节按段落二次切分
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=80
            ).split_text(part)
        docs += [
            Document(page_content=c, metadata={
                "source": filename,
                "type": doc_type,
                "title": title,
            })
            for c in chunks
        ]
    return docs


def build_documents() -> list[Document]:
    """汇总所有来源的文档。ingest 入库与混合检索共用，保证口径一致。"""
    docs = load_products()
    docs += load_markdown("after_sales_rules.md", "after_sales_rule")
    docs += load_markdown("faq.md", "faq")
    return docs
