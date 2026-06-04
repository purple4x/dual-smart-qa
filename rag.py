"""
RAG 知识库：切块 → 向量化 → 存入 ChromaDB → 检索 → 按文档回答
"""

import hashlib
import os
import re
from http import HTTPStatus

import dashscope
from dashscope import TextEmbedding

from llm import chat, get_api_key, is_api_key_configured

# 每块大约 400 字，块之间重叠 80 字，避免一句话被截断
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
TOP_K = 3  # 每次检索最相关的 3 段
EMBED_MODEL = "text-embedding-v2"
EMBED_BATCH = 25  # 通义 Embedding 单次最多约 25 条


def _import_chromadb():
    """延迟导入，并关闭遥测，减少云端依赖冲突。"""
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    import chromadb

    return chromadb


def chunk_text(text: str) -> list[str]:
    """把长文档切成多个小段，方便向量化和检索。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        # 尽量在换行处截断，少截断句子
        if end < len(text):
            last_nl = text.rfind("\n", start, end)
            if last_nl > start + CHUNK_SIZE // 2:
                end = last_nl + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """调用通义 Embedding，把文本列表转成向量列表。"""
    if not is_api_key_configured():
        raise ValueError("请先在 .env 中配置有效的 DASHSCOPE_API_KEY")
    if not texts:
        return []

    dashscope.api_key = get_api_key()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        response = TextEmbedding.call(model=EMBED_MODEL, input=batch)
        if response.status_code != HTTPStatus.OK:
            raise RuntimeError(
                f"Embedding 失败: {getattr(response, 'code', 'unknown')} - "
                f"{getattr(response, 'message', '无详情')}"
            )
        # 按 text_index 排序，保证顺序与 batch 一致
        items = sorted(response.output["embeddings"], key=lambda x: x["text_index"])
        all_embeddings.extend(item["embedding"] for item in items)

    return all_embeddings


def _collection_name(filename: str) -> str:
    """Chroma 集合名只能含字母数字下划线，用文件名 hash 生成。"""
    digest = hashlib.md5(filename.encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", filename)[:20]
    return f"doc_{safe}_{digest}"


def index_document(text: str, filename: str) -> tuple[object, int]:
    """
    建立知识库索引：切块 → 向量化 → 写入 ChromaDB。
    返回 (collection, 块数量)。
    """
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("文档内容为空，无法建立索引。")

    embeddings = embed_texts(chunks)

    chromadb = _import_chromadb()
    client = chromadb.Client()
    collection = client.create_collection(name=_collection_name(filename))
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
    )
    return collection, len(chunks)


def retrieve(collection, question: str, top_k: int = TOP_K) -> list[str]:
    """根据用户问题，检索最相关的文档片段。"""
    count = collection.count()
    if count == 0:
        return []

    # 文档片段很少时，每次带上全部内容，避免问法不同就检索偏了
    if count <= top_k:
        return collection.get()["documents"]

    query_embedding = embed_texts([question])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
    )
    documents = results.get("documents", [[]])[0]
    return [doc for doc in documents if doc]


def _retrieval_query(messages: list[dict]) -> str:
    """
    构造用于检索的 query。
    追问里常有「它 / 这个 / 内容 / 全文」，拼接上一轮问题，检索更准。
    """
    current = messages[-1]["content"]
    hints = ("它", "这个", "那份", "文档", "内容", "全文", "原文", "发给我", "输出", "给我")
    if len(messages) >= 2 and any(h in current for h in hints):
        prev_questions = [
            m["content"] for m in messages[:-1] if m["role"] == "user"
        ]
        if prev_questions:
            return f"{prev_questions[-1]} {current}"
    return current


def rag_chat(messages: list[dict], collection) -> str:
    """
    知识库问答：先检索相关片段，再让模型只根据这些片段回答。
    """
    question = messages[-1]["content"]
    search_query = _retrieval_query(messages)
    chunks = retrieve(collection, search_query)

    # 用户明确要「原文 / 全文」时，短文档直接给全部片段
    want_full = any(k in question for k in ("全文", "原文", "输出", "发给我", "内容给我", "贴出来"))
    if want_full and collection.count() <= 10:
        chunks = collection.get()["documents"]

    if not chunks:
        context = "（未检索到相关内容）"
    else:
        context = "\n\n---\n\n".join(chunks)

    system_prompt = f"""你是一个文档问答助手。请只根据下面「参考资料」回答用户问题。

规则：
1. 只能使用参考资料中的信息，禁止编造。
2. 用户要求概括、引用、复述或输出文档内容时，可以直接基于参考资料作答。
3. 只有参考资料里确实没有任何相关信息时，才回答：「根据已上传的文档，未找到相关信息。」
4. 用简洁、准确的中文回答。

参考资料：
{context}"""

    # system + 最近几轮对话（保留上下文，但不把整篇文档塞进历史）
    api_messages = [{"role": "system", "content": system_prompt}]
    api_messages.extend(messages[-6:])

    return chat(api_messages)
