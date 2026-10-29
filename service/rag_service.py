"""
RAG 服务封装

把 LangGraph 工作流包装成可编程调用的函数，供 FastAPI / Streamlit 复用。
采用惰性导入：首次调用时才加载工作流图（连接 Milvus、加载向量模型），
避免服务启动时就因为依赖未就绪而失败。
"""

_graph = None  # 缓存编译好的自适应 RAG 工作流图


def _get_graph():
    """惰性获取自适应 RAG（Adaptive RAG）工作流图，首次调用时加载并缓存"""
    global _graph
    if _graph is None:
        from graph2.graph_2 import graph
        _graph = graph
    return _graph


def ask(question: str) -> str:
    """调用自适应 RAG 工作流，返回最终回答文本"""
    if not question or not question.strip():
        return "请输入有效的问题。"
    graph = _get_graph()
    result = graph.invoke({"question": question.strip()})
    return result.get("generation", "")


def ask_stream(question: str):
    """
    流式调用自适应 RAG 工作流，逐个节点产出执行进度和最终回答。
    产出项为 dict：
        {"node": 节点名}    —— 当前执行到的节点
        {"answer": 文本}    —— 生成节点产出的最终回答
    """
    if not question or not question.strip():
        yield {"answer": "请输入有效的问题。"}
        return
    graph = _get_graph()
    for output in graph.stream({"question": question.strip()}):
        for key, value in output.items():
            yield {"node": key}
            if isinstance(value, dict) and value.get("generation"):
                yield {"answer": value["generation"]}


def ingest_documents(md_dir: str = None) -> int:
    """
    单进程入库（适合在 API 里调用，避免 Windows 下 multiprocessing 的坑）。
    命令行批量入库请用：python main.py ingest（多进程，适合海量数据）。

    注意：集合已存在时会先清空再重建。
    """
    import os

    from documents.markdown_parser import MarkdownParser
    from documents.milvus_db import MilvusVectorSave
    from utils.log_utils import log

    if md_dir is None:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        md_dir = os.path.join(root_dir, 'datas', 'md')

    mv = MilvusVectorSave()
    mv.create_collection()  # 会清空重建集合
    mv.create_connection()

    parser = MarkdownParser()
    total = 0
    for f in sorted(os.listdir(md_dir)):
        if not f.endswith('.md'):
            continue
        try:
            docs = parser.parse_markdown_to_documents(os.path.join(md_dir, f))
            if docs:
                mv.add_documents(docs)
                total += len(docs)
                log.info(f"已入库 {f}：{len(docs)} 个片段")
        except Exception as e:
            log.error(f"入库失败 {f}: {e}")
    return total
