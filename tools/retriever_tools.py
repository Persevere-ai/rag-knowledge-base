from langchain_core.tools import create_retriever_tool
from documents.milvus_db import MilvusVectorSave

mv = MilvusVectorSave()
mv.create_connection()
retriever = mv.vector_store_saved.as_retriever(
    search_type='similarity',  # 仅返回相似度超过阈值的文档
    search_kwargs={
        "k": 4,
        "score_threshold": 0.1,
        "ranker_type": "rrf",
        "ranker_params": {"k": 100},
        'filter': {"category": "content"}
    }
)


# 注意：description 描述的是知识库的业务范围，LLM 会据此判断何时调用检索工具。
# 请把它改成你自己知识库的主题（例如「公司内部制度」「医疗指南」「财务法规」等）。
retriever_tool = create_retriever_tool(
    retriever,
    'rag_retriever',
    '搜索并返回知识库中与用户问题相关的文档内容'
)