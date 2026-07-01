from rag.config import TOP_K
from rag.embedder import embed_texts


def build_retrieval_query(messages: list[dict]) -> str:
    current = messages[-1]["content"]
    hints = ("它", "这个", "那份", "文档", "内容", "全文", "原文", "发给我", "输出", "给我")
    if len(messages) >= 2 and any(h in current for h in hints):
        prev = [m["content"] for m in messages[:-1] if m["role"] == "user"]
        if prev:
            return f"{prev[-1]} {current}"
    return current


def retrieve(collection, question: str, top_k: int = TOP_K) -> list[str]:
    count = collection.count()
    if count == 0:
        return []
    if count <= top_k:
        return collection.get()["documents"]

    query_embedding = embed_texts([question])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
    )
    return [d for d in results.get("documents", [[]])[0] if d]


def retrieve_for_chat(messages: list[dict], collection) -> list[str]:
    question = messages[-1]["content"]
    chunks = retrieve(collection, build_retrieval_query(messages))
    want_full = any(k in question for k in ("全文", "原文", "输出", "发给我", "内容给我", "贴出来"))
    if want_full and collection.count() <= 10:
        chunks = collection.get()["documents"]
    return chunks
