import streamlit as st
import pandas as pd
import os, time, tempfile, re, math, uuid, itertools, io
import arxiv, requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit_agraph import agraph, Node, Edge, Config
from google import genai  # 使用最新 SDK

# ================= 1. 配置与初始化 =================
st.set_page_config(page_title="AI 深度研读助手", layout="wide", page_icon="🎓")
USER_API_KEY = st.secrets["GOOGLE_API_KEY"]
SS_API_KEY = st.secrets.get("SS_API_KEY", "")
client = genai.Client(api_key=USER_API_KEY)

# 简约 CSS
st.markdown("""<style>
    .abstract-box { background:#f0f2f6; padding:12px; border-radius:8px; border-left:5px solid #4CAF50; font-size:.9em; margin-bottom:10px; }
    .contribution-box { background:#fffbeb; border-left:4px solid #f59e0b; padding:8px; border-radius:6px; font-size:.85em; color:#78350f; margin-bottom:8px; }
    .chat-panel { height:500px; overflow-y:auto; border:1px solid #e2e8f0; padding:12px; background:#fafafa; border-radius:10px; }
    .chat-user { background:#dbeafe; padding:8px; border-radius:8px; margin:5px 0; }
    .chat-bot { background:#f0fdf4; padding:8px; border-radius:8px; margin:5px 0; }
</style>""", unsafe_allow_html=True)

# 状态管理
for k, v in {
    "search_results": [], "citations_global_cache": {}, "contributions_cache": {},
    "chat_history": [], "topics": {"默认主题": {"files": [], "chunks": [], "embeddings": []}},
    "active_topic": "默认主题", "focus_paper_id": None, "trackers": {}, "notes": []
}.items():
    if k not in st.session_state: st.session_state[k] = v

# ================= 2. 核心 API 工具 =================

def get_embedding(text):
    """使用最新 SDK 调用 text-embedding-004"""
    try:
        res = client.models.embed_content(model="text-embedding-004", contents=text)
        return res.embeddings[0].values
    except Exception as e:
        st.error(f"Embedding Error: {e}"); return None

def get_ai_response(prompt, context=""):
    """使用最新 SDK 调用 Gemini 1.5 Pro"""
    full_prompt = f"Context: {context}\n\nQuestion: {prompt}" if context else prompt
    try:
        response = client.models.generate_content(model="gemini-1.5-pro", contents=full_prompt)
        return response.text
    except Exception as e: return f"Error: {e}"

def get_contribution(title, abstract):
    key = title[:60]
    if key in st.session_state.contributions_cache: return st.session_state.contributions_cache[key]
    prompt = f"用一句话总结这篇论文的核心创新点（40字内）：\n标题：{title}\n摘要：{abstract}"
    res = client.models.generate_content(model="gemini-1.5-flash", contents=prompt).text.strip()
    st.session_state.contributions_cache[key] = res
    return res

# ================= 3. 文献与主题处理 =================

def process_pdf(file_path, file_name, topic_name):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    t = st.session_state.topics[topic_name]
    loader = PyPDFLoader(file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(loader.load())
    
    with st.spinner(f"正在向量化 {file_name}..."):
        for chunk in chunks:
            vec = get_embedding(chunk.page_content)
            if vec:
                t["chunks"].append({"text": chunk.page_content, "source": file_name})
                t["embeddings"].append(vec)
    if file_name not in t["files"]: t["files"].append(file_name)

def vector_search(query, topic_name, top_k=5):
    t = st.session_state.topics[topic_name]
    if not t["embeddings"]: return []
    q_vec = get_embedding(query)
    if not q_vec: return []
    
    import numpy as np
    # 简单的余弦相似度计算，替代复杂的 FAISS 以精简代码
    sims = [np.dot(q_vec, v) / (np.linalg.norm(q_vec) * np.linalg.norm(v)) for v in t["embeddings"]]
    top_indices = np.argsort(sims)[-top_k:][::-1]
    return [t["chunks"][i] for i in top_indices]

# ================= 4. 侧边栏 =================
with st.sidebar:
    st.title("🎓 研读助手")
    st.subheader("🗂️ 主题管理")
    st.session_state.active_topic = st.selectbox("当前主题", list(st.session_state.topics.keys()))
    
    with st.expander("➕ 新建/上传"):
        new_tn = st.text_input("新主题名")
        if st.button("创建") and new_tn:
            st.session_state.topics[new_tn] = {"files":[], "chunks":[], "embeddings":[]}
            st.rerun()
        up_file = st.file_uploader("上传 PDF", type="pdf")
        if up_file and st.button("确认解析"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(up_file.getvalue())
                process_pdf(tmp.name, up_file.name, st.session_state.active_topic)
            st.success("解析完成"); st.rerun()

# ================= 5. 主界面 =================
tab1, tab2, tab3 = st.tabs(["🔍 检索图谱", "📖 深度研读", "📌 追踪笔记"])

with tab1:
    col_s1, col_s2 = st.columns([4, 1])
    with col_s1: q = st.text_input("关键词搜索 ArXiv", placeholder="例如: zoomorphic robots education")
    with col_s2: s_btn = st.button("搜索", use_container_width=True)
    
    if s_btn and q:
        search = arxiv.Search(query=q, max_results=20, sort_by=arxiv.SortCriterion.Relevance)
        st.session_state.search_results = list(search.results())
    
    for i, res in enumerate(st.session_state.search_results):
        with st.expander(f"{res.title} ({res.published.year})"):
            st.write(f"作者: {', '.join([a.name for a in res.authors])}")
            st.markdown(f'<div class="contribution-box">✨ {get_contribution(res.title, res.summary)}</div>', unsafe_allow_html=True)
            st.write(res.summary)
            if st.button("⬇️ 载入主题", key=f"load_{i}"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    res.download_pdf(dirpath=tempfile.gettempdir(), filename=f"tmp_{i}.pdf")
                    process_pdf(os.path.join(tempfile.gettempdir(), f"tmp_{i}.pdf"), res.title, st.session_state.active_topic)
                st.toast("已加入库")

with tab2:
    st.subheader(f"💬 对话: {st.session_state.active_topic}")
    chat_container = st.container()
    with chat_container:
        for m in st.session_state.chat_history:
            role_class = "chat-user" if m["role"] == "user" else "chat-bot"
            st.markdown(f'<div class="{role_class}">{m["content"]}</div>', unsafe_allow_html=True)
            
    if prompt := st.chat_input("基于已加载文献提问..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        # RAG 检索
        related_chunks = vector_search(prompt, st.session_state.active_topic)
        context = "\n".join([f"Source: {c['source']}\nContent: {c['text']}" for c in related_chunks])
        answer = get_ai_response(prompt, context)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

with tab3:
    col_n1, col_n2 = st.columns(2)
    with col_n1:
        st.subheader("🔔 关键词追踪")
        tkw = st.text_input("添加追踪词")
        if st.button("添加") and tkw:
            st.session_state.trackers[tkw] = datetime.now()
            st.toast("已开启追踪")
        for k in st.session_state.trackers: st.write(f"· {k}")
    with col_n2:
        st.subheader("📌 随手记")
        note = st.text_area("记录灵感...")
        if st.button("保存笔记") and note:
            st.session_state.notes.append({"text": note, "time": datetime.now().strftime("%Y-%m-%d %H:%M")})
        for n in reversed(st.session_state.notes):
            st.info(f"{n['time']}: {n['text']}")
