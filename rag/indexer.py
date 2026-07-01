import hashlib
import os
import re

from rag.chunker import chunk_text
from rag.embedder import embed_texts


def _import_chromadb():
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    import chromadb

    return chromadb


def _collection_name(filename: str) -> str:
    digest = hashlib.md5(filename.encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", filename)[:20]
    return f"doc_{safe}_{digest}"


def index_document(text: str, filename: str) -> tuple[object, int]:
    """建立知识库索引：切块 → 向量化 → 写入 ChromaDB。"""
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("文档内容为空，无法建立索引。")

    embeddings = embed_texts(chunks)

    chromadb = _import_chromadb()
    client = chromadb.Client()
    name = _collection_name(filename)
    try:
        client.delete_collection(name)
    except Exception:
        pass

    collection = client.create_collection(name=name)
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
    )
    return collection, len(chunks)
