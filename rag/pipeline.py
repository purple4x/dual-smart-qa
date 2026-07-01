"""
RAG 主流程：检索 → 生成初稿 → Self-Reflection（方案 A 纯 Prompt 自检）→ 最终答案
"""

from dataclasses import dataclass, field

from llm import chat
from rag.config import NOT_FOUND_REPLY
from rag.indexer import index_document
from rag.reflection import ReflectionResult, reflect_answer
from rag.retriever import retrieve_for_chat

__all__ = ["index_document", "rag_chat", "rag_chat_detailed", "RagResult"]


@dataclass
class RagResult:
    answer: str
    chunks: list[str] = field(default_factory=list)
    draft_answer: str = ""
    reflection: ReflectionResult | None = None


def _build_context(chunks: list[str]) -> str:
    if not chunks:
        return "（未检索到相关内容）"
    return "\n\n---\n\n".join(chunks)


def _generate_draft(messages: list[dict], context: str) -> str:
    system_prompt = f"""你是文档问答助手。只根据「参考资料」回答。

规则：
1. 禁止编造参考资料中没有的信息。
2. 可概括、引用、复述参考资料。
3. 确实无相关信息时，回答：「{NOT_FOUND_REPLY}」

参考资料：
{context}"""

    api_messages = [{"role": "system", "content": system_prompt}]
    api_messages.extend(messages[-6:])
    return chat(api_messages)


def rag_chat_detailed(messages: list[dict], collection) -> RagResult:
    question = messages[-1]["content"]
    chunks = retrieve_for_chat(messages, collection)
    context = _build_context(chunks)

    draft = _generate_draft(messages, context)
    # 方案 A：纯 Prompt 语义自检，reflection 内不调用 Embedding
    reflection = reflect_answer(question, context, draft)

    return RagResult(
        answer=reflection.final_answer,
        chunks=chunks,
        draft_answer=draft,
        reflection=reflection,
    )


def rag_chat(messages: list[dict], collection) -> str:
    return rag_chat_detailed(messages, collection).answer
