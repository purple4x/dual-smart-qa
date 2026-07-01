"""dual-smart-qa RAG 包（含 Self-Reflection 方案 A）。"""

from rag.indexer import index_document
from rag.pipeline import RagResult, rag_chat, rag_chat_detailed
from rag.reflection import ReflectionResult, reflect_answer

__all__ = [
    "index_document",
    "rag_chat",
    "rag_chat_detailed",
    "RagResult",
    "reflect_answer",
    "ReflectionResult",
]
