"""
文档解析：把上传的文件转成纯文本。
支持：.txt、.pdf、.docx
"""

from io import BytesIO

from docx import Document
from pypdf import PdfReader


def load_txt(file_bytes: bytes) -> str:
    """读取 TXT，尝试 utf-8，失败则用 gbk（常见中文 Windows 编码）。"""
    for encoding in ("utf-8", "gbk"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别 TXT 文件编码，请另存为 UTF-8 后重试。")


def load_pdf(file_bytes: bytes) -> str:
    """读取 PDF，按页拼接文本。"""
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    if not pages:
        raise ValueError("PDF 中未提取到文字（可能是扫描版图片 PDF）。")
    return "\n\n".join(pages)


def load_docx(file_bytes: bytes) -> str:
    """读取 Word (.docx)，拼接段落。"""
    doc = Document(BytesIO(file_bytes))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        raise ValueError("Word 文档中没有可读文字。")
    return "\n\n".join(paragraphs)


def load_document(filename: str, file_bytes: bytes) -> str:
    """
    根据文件扩展名，调用对应的解析函数。
    filename: 用户上传时的文件名，如 "报告.pdf"
    """
    name = filename.lower()
    if name.endswith(".txt"):
        return load_txt(file_bytes)
    if name.endswith(".pdf"):
        return load_pdf(file_bytes)
    if name.endswith(".docx"):
        return load_docx(file_bytes)
    raise ValueError("仅支持 .txt、.pdf、.docx 格式。")
