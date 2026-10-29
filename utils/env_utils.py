"""
环境配置读取模块

所有敏感信息（API Key、数据库地址等）优先从项目根目录的 `.env` 文件读取，
`.env` 不存在时回退到系统环境变量。

首次使用建议：复制 `.env.example` 为 `.env`，把 API Key 填进去。
下方默认值均为「通用/本地」占位，真实地址请通过 `.env` 配置（.env 不会被提交到 Git）。
"""

import os

from dotenv import load_dotenv

# 读取 .env 文件（override=True 表示 .env 中的值优先于系统环境变量）
load_dotenv(override=True)

# ============ 大模型（LLM）配置 ============
# API Key：优先 .env，其次系统环境变量
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
# 接口地址：OpenAI 官方（或任意 OpenAI 兼容的第三方中转，如 DeepSeek / 通义千问）
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
# 模型名称
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

# DeepSeek 专用 Key（可选）
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')

# ============ 网络搜索工具（Tavily）配置 ============
# graph2（自适应 RAG）的"网络搜索"分支需要；不需要可留空
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY', '')

# ============ 向量数据库（Milvus）配置 ============
# 默认指向本地单机版（docker compose up -d 启动后即为 localhost:19530）
MILVUS_URI = os.getenv('MILVUS_URI', 'http://localhost:19530')
# 集合名（可自定义）
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 't_collection01')

# ============ Embedding（向量化）模型配置 ============
# 本地开源中文向量模型；换成其它模型时需同步修改 milvus_db.py 中 dense 字段的维度
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'BAAI/bge-small-zh-v1.5')
# 计算设备：cpu 或 cuda
EMBEDDING_DEVICE = os.getenv('EMBEDDING_DEVICE', 'cpu')
