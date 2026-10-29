"""
Embedding（向量化）模型封装模块

提供两种向量化方案：
1. openai_embedding —— 调用 OpenAI 兼容接口的云端向量模型（用于语义切割 SemanticChunker）
2. bge_embedding     —— 本地开源中文向量模型 BGE（用于向量知识库的入库与检索）

模型名和计算设备在 .env 中配置，请勿在此硬编码。
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from utils.env_utils import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
)

# 云端向量模型（OpenAI 兼容接口）
openai_embedding = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_BASE_URL,
)

# 本地开源向量模型（BGE 系列，中文检索效果好）
# 注意：若更换模型，需同步修改 documents/milvus_db.py 中 dense 字段的维度 dim。
# bge-small-zh-v1.5 -> 512 维；bge-large-zh-v1.5 -> 1024 维
bge_embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": EMBEDDING_DEVICE},
    encode_kwargs={"normalize_embeddings": True},  # 归一化，便于用内积近似余弦相似度
)
