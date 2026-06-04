"""
通义千问 API 封装。
本文件只负责：读 API Key、把对话发给模型、拿回回复。
"""

import os
from http import HTTPStatus

import dashscope
from dashscope import Generation
from dotenv import load_dotenv

# 从项目根目录的 .env 加载环境变量
load_dotenv()

# 模型名可在阿里云控制台查看；turbo 便宜、适合练手
MODEL = "qwen-turbo"


def get_api_key() -> str | None:
    """
    读取 API Key，优先级：
    1. 环境变量（本地 .env 或 Streamlit Cloud Secrets）
    2. st.secrets（Streamlit Cloud 备用）
    """
    key = os.getenv("DASHSCOPE_API_KEY")
    if key and key.strip() and key != "your_api_key_here":
        return key.strip()

    try:
        import streamlit as st

        if "DASHSCOPE_API_KEY" in st.secrets:
            return str(st.secrets["DASHSCOPE_API_KEY"]).strip()
    except Exception:
        pass

    return None


def is_api_key_configured() -> bool:
    key = get_api_key()
    return bool(key and key.strip() and key != "your_api_key_here")


def chat(messages: list[dict]) -> str:
    """
    多轮对话：把历史消息发给通义千问，返回助手回复文本。

    messages 格式示例：
        [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，有什么可以帮你？"},
            {"role": "user", "content": "介绍一下 Python"},
        ]
    """
    if not is_api_key_configured():
        raise ValueError(
            "未配置 API Key。本地请在 .env 填写 DASHSCOPE_API_KEY；"
            "线上请在 Streamlit Cloud → Settings → Secrets 中配置。"
        )

    dashscope.api_key = get_api_key()

    response = Generation.call(
        model=MODEL,
        messages=messages,
        result_format="message",
    )

    if response.status_code == HTTPStatus.OK:
        return response.output.choices[0].message.content

    raise RuntimeError(
        f"API 调用失败: {getattr(response, 'code', 'unknown')} - "
        f"{getattr(response, 'message', '无详情')}"
    )
