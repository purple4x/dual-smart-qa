from http import HTTPStatus

import dashscope
from dashscope import TextEmbedding

from llm import get_api_key, is_api_key_configured
from rag.config import EMBED_BATCH, EMBED_MODEL


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
        items = sorted(response.output["embeddings"], key=lambda x: x["text_index"])
        all_embeddings.extend(item["embedding"] for item in items)

    return all_embeddings
