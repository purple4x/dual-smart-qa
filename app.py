import streamlit as st

from document_loader import load_document
from llm import chat, is_api_key_configured
from rag import index_document, rag_chat

st.set_page_config(page_title="双模智能问答", page_icon="💬", layout="wide")

# ---------- 初始化 session（刷新页面会清空）----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_text" not in st.session_state:
    st.session_state.document_text = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None
if "collection" not in st.session_state:
    st.session_state.collection = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

has_document = st.session_state.document_text is not None

st.title("双模智能问答系统")
if has_document:
    st.caption(f"当前模式：知识库问答（已加载：{st.session_state.document_name}）")
else:
    st.caption("当前模式：普通聊天（未上传文档）")

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("文档")
    uploaded = st.file_uploader(
        "上传 PDF / Word / TXT",
        type=["pdf", "docx", "txt"],
        help="上传后自动建立知识库，只按文档内容回答。",
    )

    if uploaded is not None:
        if uploaded.name != st.session_state.document_name:
            try:
                with st.spinner("正在解析文档…"):
                    text = load_document(uploaded.name, uploaded.getvalue())
                with st.spinner("正在切块并向量化，请稍候…"):
                    collection, chunk_count = index_document(text, uploaded.name)

                st.session_state.document_text = text
                st.session_state.document_name = uploaded.name
                st.session_state.collection = collection
                st.session_state.chunk_count = chunk_count
                st.session_state.messages = []
                st.success(f"已加载：{uploaded.name}（{chunk_count} 个片段）")
            except Exception as e:
                st.error(str(e))

    if has_document:
        st.info(f"📄 {st.session_state.document_name}")
        st.metric("提取字符数", len(st.session_state.document_text))
        st.metric("知识库片段数", st.session_state.chunk_count)
        with st.expander("预览文档开头"):
            preview = st.session_state.document_text[:800]
            if len(st.session_state.document_text) > 800:
                preview += "\n\n…（后面还有内容）"
            st.text(preview)
        if st.button("移除文档"):
            st.session_state.document_text = None
            st.session_state.document_name = None
            st.session_state.collection = None
            st.session_state.chunk_count = 0
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.header("设置")
    if is_api_key_configured():
        st.success("API Key 已配置")
    else:
        st.error("未检测到 API Key")
        st.markdown("请复制 `.env.example` 为 `.env`，填入通义千问 Key。")

    if st.button("清空对话"):
        st.session_state.messages = []
        st.rerun()

# ---------- 对话历史 ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------- 用户输入 ----------
if prompt := st.chat_input("输入你的问题…"):
    if not is_api_key_configured():
        st.error("请先在 .env 中配置 DASHSCOPE_API_KEY 后刷新页面。")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("思考中…"):
            try:
                if has_document and st.session_state.collection is not None:
                    reply = rag_chat(st.session_state.messages, st.session_state.collection)
                else:
                    reply = chat(st.session_state.messages)
            except Exception as e:
                st.error(str(e))
                st.session_state.messages.pop()
                st.stop()
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
