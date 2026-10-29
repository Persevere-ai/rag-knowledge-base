"""
Streamlit 可视化问答界面

启动方式：
    streamlit run streamlit_app.py
"""

import streamlit as st

from service.rag_service import ask

st.set_page_config(page_title="RAG 企业知识库问答", layout="wide")

st.title("RAG 企业知识库问答")
st.caption("基于 RAG + LangGraph + Milvus 的企业知识库问答系统")

# 侧边栏说明
with st.sidebar:
    st.header("使用说明")
    st.markdown(
        """
        - 在下方输入框提问，系统会先从向量知识库检索相关文档，再交给大模型生成回答。
        - 工作流采用「自适应 RAG」：自动路由（知识库 / 网络搜索）、相关性评分、查询改写、幻觉检测。
        - 知识库外的通用问题会路由到网络搜索（需配置 Tavily Key）。
        """
    )
    st.divider()
    if st.button("清空对话历史"):
        st.session_state.messages = []
        st.rerun()

# 初始化会话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 展示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入框
if question := st.chat_input("请输入你的问题"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("正在检索并生成回答..."):
            answer = ask(question)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
