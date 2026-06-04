# dual-smart-qa

双模智能问答系统 — 实习简历项目

## 在线演示

部署完成后在此填写 Streamlit Cloud 链接，例如：  
`https://dual-smart-qa-xxxx.streamlit.app`

## 两种模式

| 模式 | 触发条件 | 行为 |
|------|----------|------|
| 普通聊天 | 未上传文档 | 直接调用通义千问，自由对话 |
| 知识库问答 | 上传 PDF / Word / TXT | 基于文档内容回答，不编造 |

## 技术栈

- **前端**：Streamlit
- **大模型**：阿里云通义千问（DashScope API）
- **文档解析**：PyPDF / python-docx
- **向量检索**：ChromaDB + 通义 Embedding
- **部署**：GitHub + Streamlit Community Cloud

## 项目结构

```
dual-smart-qa/
├── app.py              # 主入口（Streamlit 界面）
├── llm.py              # 通义千问对话
├── document_loader.py  # PDF / Word / TXT 解析
├── rag.py              # RAG 切块、向量化、检索
├── requirements.txt    # 依赖
└── .env.example        # 本地 API Key 模板
```

## 本地运行

```bash
# 1. 创建虚拟环境（Python 3.11）
python -m venv .venv
.venv\Scripts\activate        # Windows

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key
copy .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY

# 4. 启动
streamlit run app.py
```

## 部署到 Streamlit Cloud

### 第一步：推送到 GitHub

1. 在 GitHub 新建仓库 `dual-smart-qa`（**不要**勾选 README，避免冲突）
2. 在项目目录执行：

```powershell
cd D:\Projects\dual-smart-qa
git init
git add .
git commit -m "Initial commit: dual-mode QA with RAG"
git branch -M main
git remote add origin https://github.com/你的用户名/dual-smart-qa.git
git push -u origin main
```

> ⚠️ 切勿提交 `.env` 或含真实 Key 的文件（已在 `.gitignore` 中）。

### 第二步：Streamlit Cloud 部署

1. 打开 https://share.streamlit.io ，用 GitHub 登录
2. 点击 **New app**
3. 选择仓库 `dual-smart-qa`，分支 `main`，主文件 `app.py`
4. 点击 **Advanced settings**，Python 版本选 **3.11**
5. 在 **Secrets** 中填入：

```toml
DASHSCOPE_API_KEY = "sk-你的通义千问Key"
```

6. 点击 **Deploy**，等待 2～5 分钟

### 第三步：验证

- 侧边栏显示 **API Key 已配置**
- 普通聊天可用
- 上传文档后可按文档问答

## 环境要求

- Python 3.10 或 3.11
- 阿里云 DashScope API Key（需开通 `qwen-turbo` 与 `text-embedding-v2`）
