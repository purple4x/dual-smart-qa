"""RAG 模块全局配置。"""

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
TOP_K = 3
EMBED_MODEL = "text-embedding-v2"
EMBED_BATCH = 25
MAX_REFLECTION_ROUNDS = 1

NOT_FOUND_REPLY = "根据已上传的文档，未找到相关信息。"
