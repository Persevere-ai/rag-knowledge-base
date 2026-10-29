"""
FastAPI 服务：把 RAG 工作流发布为 REST API

启动方式：
    python api.py
    # 或
    uvicorn api:app --host 0.0.0.0 --port 8000

接口一览：
    GET  /health       健康检查
    POST /chat         问答（非流式，返回完整回答）
    POST /chat/stream  问答（SSE 流式，返回执行进度和回答）
    POST /ingest       入库 datas/md 下的文档（单进程）
"""

import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from service.rag_service import ask, ask_stream, ingest_documents

app = FastAPI(title="RAG 企业知识库 API", version="1.0.0")


class ChatRequest(BaseModel):
    question: str


class IngestRequest(BaseModel):
    md_dir: str = ""  # 留空则用项目下 datas/md


@app.get("/health")
def health():
    """健康检查"""
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    """非流式问答，返回完整回答"""
    answer = ask(req.question)
    return {"question": req.question, "answer": answer}


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """SSE 流式问答，逐节点返回执行进度和最终回答"""

    def gen():
        for item in ask_stream(req.question):
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/ingest")
def ingest(req: IngestRequest):
    """入库文档（单进程，注意会清空重建集合）"""
    md_dir = req.md_dir or None
    try:
        count = ingest_documents(md_dir)
        return {"status": "ok", "ingested": count}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
