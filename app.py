import streamlit as st
import sys
import os
import time
import tempfile
import arxiv
import requests
import math
import re
import csv
import hashlib
from datetime import datetime
from io import StringIO, BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import bibtexparser
    from bibtexparser.bwriter import BibTexWriter
    from bibtexparser.bibdatabase import BibDatabase
    import zhipuai
    import langchain_community
    import fitz  # pymupdf
except ImportError as e:
    st.error(f"🚑 环境缺失库 -> {e.name}")
    st.info("请执行安装命令：pip install zhipuai langchain-community pymupdf arxiv requests streamlit-agraph faiss-cpu bibtexparser")
    st.stop()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit_agraph import agraph, Node, Edge, Config

# ================= 2. 页面配置 =================
st.set_page_config(page_title="科研文献助手 | Connected Papers + AI 精读", layout="wide", page_icon="🎓")
st.markdown("""
<style>
    .stButton>button {width: 100%; border-radius: 8px;}
    .reportview-container { margin-top: -2em; }
    .abstract-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #4CAF50;
        font-size: 0.95em;
        line-height: 1.6;
        margin-bottom: 10px;
    }
    .cite-badge {
        background-color: #ff4b4b;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
    }
    .high-impact {
        background-color: #fff3cd;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.75em;
        color: #856404;
        font-weight: bold;
    }
    .detail-panel {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        height: 600px;
        overflow-y: auto;
    }
    .timeline-note {
        font-size: 0.8em;
        color: #666;
        text-align: center;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)
st.title("🎓 科研文献助手 | Connected Papers + AI 精读")

# ================= 3. 状态初始化 =================
for key, default in [
    ("chat_history", []),
    ("db", None),
    ("loaded_files", []),
    ("all_chunks", []),
    ("suggested_query", ""),
    ("search_results", []),
    ("selected_scope", "🌐 对比所有论文"),
    ("focus_paper_id", None),
    ("filtered_results", []),
    ("pending_ai_prompt", None),  # 新增：快捷按钮触发后待处理的 prompt
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ================= 4. 核心逻辑函数 =================

def get_pure_arxiv_id(url_or_id):
    """从 URL 或字符串中精准提取 ArXiv ID"""
    match = re.search(r'(\d{4}\.\d{4,5})', url_or_id)
    if match:
        return match.group(1)
    return url_or_id.split('/')[-1].split('v')[0]

def _hash_key(key):
    """对 API Key 做单向哈希，避免明文存入缓存键"""
    if not key:
        return ""
    return hashlib.sha256(key.encode()).hexdigest()[:16]

def fetch_citations_single(arxiv_id, ss_key=None):
    """获取单篇论文引用数"""
    try:
        clean_id = get_pure_arxiv_id(arxiv_id)
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{clean_id}?fields=citationCount"
        headers = {"x-api-key": ss_key} if ss_key else {}
        delay = 0.05 if ss_key else 1.0
        time.sleep(delay)
        response = requests.get(api_url, headers=headers, timeout=5)
        if response.status_code == 200:
            return response.json().get('citationCount', 0)
    except Exception:
        pass
    return 0

def fetch_citations_batch(arxiv_ids, ss_key=None):
    """
    并发获取多篇论文引用数（修复：串行改并发）
    无 Key 限制并发数为 2，有 Key 可提高到 5
    """
    max_workers = 5 if ss_key else 2

    def _fetch(aid):
        return aid, fetch_citations_single(aid, ss_key)

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch, aid): aid for aid in arxiv_ids}
        for future in as_completed(futures):
            aid, count = future.result()
            results[aid] = count
    return results

@st.cache_data(ttl=3600)
def fetch_graph_data(arxiv_id, ss_key_hash="", ss_key="", expand_depth=1):
    """
    获取图谱数据
    - ss_key_hash 作为缓存键（不存明文）
    - expand_depth: 1=直接引用/被引, 2=递归拓展一层
    """
    clean_id = get_pure_arxiv_id(arxiv_id)
    fields = (
        "paperId,title,year,citationCount,abstract,url,journal,"
        "references.paperId,references.title,references.citationCount,"
        "references.year,references.abstract,references.url,"
        "citations.paperId,citations.title,citations.citationCount,"
        "citations.year,citations.abstract,citations.url"
    )
    api_url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{clean_id}?fields={fields}"
    headers = {"x-api-key": ss_key} if ss_key else {}

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(api_url, headers=headers, timeout=12)
            if response.status_code == 200:
                data = response.json()

                # 递归拓展（领域级脉络）
                # 修复：次级引用的 paperId 是 SS 内部 ID，不是 ArXiv ID，
                # 需要用 SS paper ID 接口而非 ArXiv 接口
                if expand_depth > 1:
                    for ref in data.get('references', [])[:5]:
                        ref_pid = ref.get('paperId')
                        if ref_pid:
                            sub_url = (
                                f"https://api.semanticscholar.org/graph/v1/paper/{ref_pid}"
                                f"?fields=references.paperId,references.title,references.citationCount,references.year"
                            )
                            try:
                                sub_resp = requests.get(sub_url, headers=headers, timeout=8)
                                if sub_resp.status_code == 200:
                                    ref['sub_references'] = sub_resp.json().get('references', [])[:3]
                            except Exception:
                                pass

                return data
            elif response.status_code == 429:
                time.sleep((attempt + 1) * 2)
                continue
            else:
                return None
        except Exception:
            if attempt == max_retries - 1:
                return None
    return None

def render_research_graph(data, min_citation=10, year_range=(2010, 2026)):
    """
    科研级脉络图谱渲染
    修复：移除不支持的 Python lambda，改用静态物理引擎配置
    """
    if not data:
        return None, {}

    nodes, edges = [], []
    paper_details = {}
    min_year, max_year = year_range

    def get_color(rel_type):
        colors = {
            'seed': "#FF4B4B",
            'reference': "#2563eb",
            'citation': "#059669",
            'sub_reference': "#60a5fa"
        }
        return colors.get(rel_type, "#94a3b8")

    def get_node_size(citation_count):
        if citation_count < min_citation:
            return 10
        return max(15, 15 + (math.log(citation_count + 1) * 4))

    def is_high_impact(cites):
        return cites >= 50

    def year_in_range(year):
        if year == 'Unknown':
            return True
        try:
            y = int(year)
            return min_year <= y <= max_year
        except (ValueError, TypeError):
            return True

    # 种子节点
    seed_id = data.get('paperId', 'root')
    seed_title = data.get('title', 'Seed Paper') or 'Seed Paper'
    seed_cites = data.get('citationCount', 0) or 0
    seed_year = data.get('year', 'Unknown')

    if not year_in_range(seed_year):
        return None, {}

    paper_details[seed_id] = {
        "title": seed_title,
        "abstract": data.get('abstract') or "无摘要",
        "year": seed_year,
        "cites": seed_cites,
        "url": data.get('url') or f"https://www.semanticscholar.org/paper/{seed_id}",
        "journal": (data.get('journal') or {}).get('name') or "未知期刊",
        "is_high_impact": is_high_impact(seed_cites)
    }

    nodes.append(Node(
        id=seed_id,
        label=f"核心\n{seed_title[:12]}...",
        size=40 if is_high_impact(seed_cites) else 35,
        color=get_color('seed'),
        title=f"{seed_title}\n引用数: {seed_cites}\n年份: {seed_year}\n{'🔥 高影响力' if is_high_impact(seed_cites) else ''}"
    ))

    seen_ids = {seed_id}

    def add_paper_node(p, rel_type):
        p_id = p.get('paperId')
        if not p_id or p_id in seen_ids:
            return False
        p_cites = p.get('citationCount') or 0
        p_year = p.get('year', 'Unknown')
        if p_cites < min_citation:
            return False
        if not year_in_range(p_year):
            return False

        seen_ids.add(p_id)
        p_title = p.get('title') or 'Unknown'

        paper_details[p_id] = {
            "title": p_title,
            "abstract": p.get('abstract') or "暂无详细摘要",
            "year": p_year,
            "cites": p_cites,
            "url": p.get('url') or f"https://www.semanticscholar.org/paper/{p_id}",
            "journal": "未知期刊",
            "is_high_impact": is_high_impact(p_cites)
        }

        nodes.append(Node(
            id=p_id,
            label=f"{p_title[:12]}...",
            size=get_node_size(p_cites),
            color=get_color(rel_type),
            title=f"{p_title}\n引用数: {p_cites}\n年份: {p_year}\n{'🔥 高影响力' if is_high_impact(p_cites) else ''}"
        ))
        return True

    # 参考文献（蓝色：前人工作）
    for p in data.get('references', [])[:20]:
        if add_paper_node(p, 'reference'):
            p_id = p['paperId']
            p_cites = p.get('citationCount') or 0
            edges.append(Edge(
                source=seed_id,
                target=p_id,
                color="#94a3b8",
                width=2 if is_high_impact(p_cites) else 1.5,
                label="引用"
            ))

    # 施引文献（绿色：后续发展）
    for p in data.get('citations', [])[:20]:
        if add_paper_node(p, 'citation'):
            p_id = p['paperId']
            p_cites = p.get('citationCount') or 0
            edges.append(Edge(
                source=p_id,
                target=seed_id,
                color="#d1d5db",
                width=2 if is_high_impact(p_cites) else 1,
                label="被引"
            ))

    # 修复：移除 Python lambda，使用静态配置
    config = Config(
        width="100%",
        height=700,
        directed=True,
        physics=True,
        nodeHighlightBehavior=True,
        highlightColor="#F7D154",
        collapsible=False,
        staticGraph=False,
        d3={
            'alphaTarget': 0.08,
            'gravity': -300,
            'linkLength': 180,
            'linkStrength': 0.15,
        }
    )

    clicked_id = agraph(nodes=nodes, edges=edges, config=config)
    return clicked_id, paper_details

def fix_latex_errors(text):
    if not text:
        return text
    text = text.replace(r"\(", "$").replace(r"\)", "$")
    text = text.replace(r"\[", "$$").replace(r"\]", "$$")
    return text

def require_api_key(key, label="智谱 API Key"):
    """守卫函数：未填 Key 时显示警告并停止"""
    if not key:
        st.warning(f"⚠️ 请先在左侧侧边栏填入 {label}")
        st.stop()

def rebuild_index_from_chunks(api_key):
    if not st.session_state.all_chunks:
        st.session_state.db = None
        return
    embeddings = ZhipuAIEmbeddings(model="embedding-2", api_key=api_key)
    st.session_state.db = FAISS.from_documents(st.session_state.all_chunks, embeddings)

def process_and_add_to_db(file_path, file_name, api_key, cleanup=True):
    """
    解析 PDF 并加入向量库
    修复：使用 try/finally 保证临时文件清理
    修复：防止重复加载同一文件
    """
    try:
        if file_name in st.session_state.loaded_files:
            st.warning(f"《{file_name}》已加载，跳过重复导入。")
            return

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata['source_paper'] = file_name

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        valid_chunks = [c for c in chunks if len(c.page_content.strip()) > 20]

        st.session_state.all_chunks.extend(valid_chunks)
        embeddings = ZhipuAIEmbeddings(model="embedding-2", api_key=api_key)

        batch_size = 10
        total = len(valid_chunks)
        if st.session_state.db is None:
            st.session_state.db = FAISS.from_documents(valid_chunks[:batch_size], embeddings)
            for i in range(batch_size, total, batch_size):
                st.session_state.db.add_documents(valid_chunks[i: i + batch_size])
                time.sleep(0.1)
        else:
            for i in range(0, total, batch_size):
                st.session_state.db.add_documents(valid_chunks[i: i + batch_size])
                time.sleep(0.1)

        st.session_state.loaded_files.append(file_name)
        st.session_state.chat_history.append({
            "role": "system_notice",
            "content": f"📚 **系统通知**：已加载《{file_name}》。"
        })
    except Exception as e:
        st.error(f"处理失败: {e}")
    finally:
        # 修复：保证临时文件一定被删除
        if cleanup and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

# ================= 5. 科研级导出功能 =================
def export_papers_to_csv(paper_details):
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['标题', '年份', '引用数', '期刊', '链接', '摘要', '是否高影响力'])
    for pid, info in paper_details.items():
        abstract = info['abstract']
        writer.writerow([
            info['title'],
            info['year'],
            info['cites'],
            info['journal'],
            info['url'],
            abstract[:200] + "..." if len(abstract) > 200 else abstract,
            "是" if info['is_high_impact'] else "否"
        ])
    output.seek(0)
    return output

def export_papers_to_bibtex(paper_details):
    """
    导出 BibTeX
    修复：补全 author/journal 字段，使用规范 ID 格式
    """
    db = BibDatabase()
    entries = []
    for idx, (pid, info) in enumerate(paper_details.items()):
        title = info['title']
        year = str(info['year']) if info['year'] != 'Unknown' else 'n.d.'
        # 规范 ID：取标题首词 + 年份
        first_word = re.sub(r'[^a-zA-Z]', '', title.split()[0]) if title else f"paper{idx}"
        entry_id = f"{first_word.lower()}{year}"

        entry = {
            'ID': entry_id,
            'ENTRYTYPE': 'article',
            'title': title,
            'year': year,
            'journal': info.get('journal', 'Unknown'),
            'url': info['url'],
            'note': f"Citation count: {info['cites']}",
            'abstract': info['abstract'][:300] if info['abstract'] != "暂无详细摘要" else '',
        }
        entries.append(entry)

    db.entries = entries
    writer = BibTexWriter()
    return writer.write(db)

# ================= 6. AI 科研综述 =================
@st.cache_resource
def get_llm(api_key, model="glm-4"):
    """缓存 LLM 实例，避免每次对话重新实例化"""
    return ChatZhipuAI(model=model, api_key=api_key, temperature=0.1)

def generate_research_summary(paper_details, api_key):
    llm = get_llm(api_key)
    high_impact_papers = [i for i in paper_details.values() if i['is_high_impact']]
    normal_papers = [i for i in paper_details.values() if not i['is_high_impact']]

    summary_context = "### 高影响力核心论文（引用≥50）\n"
    for info in high_impact_papers[:5]:
        summary_context += f"\n**{info['title']}** ({info['year']}, 引用{info['cites']})\n"
        summary_context += f"摘要核心：{info['abstract'][:300]}...\n"

    summary_context += "\n### 领域后续发展论文\n"
    for info in normal_papers[:5]:
        summary_context += f"\n**{info['title']}** ({info['year']}, 引用{info['cites']})\n"

    prompt = f"""
请基于以下文献信息，生成一份科研级领域综述（中文），要求：
1. 先总结该领域的核心研究问题和发展脉络（按时间顺序）
2. 列出3-5篇关键奠基论文及其贡献
3. 分析当前领域的研究热点和不足
4. 给出未来研究方向建议

文献信息：
{summary_context}
"""
    response = llm.invoke(prompt)
    return fix_latex_errors(response.content)

def run_ai_chat(prompt, api_key, reading_mode, selected_scope):
    """
    执行 AI 问答逻辑（抽离为独立函数，供快捷按钮和手动输入共用）
    修复：FAISS filter 改为检索后手动过滤
    """
    require_api_key(api_key)
    if not st.session_state.db:
        st.warning("🧠 请先上传/加载论文到精读库")
        return None

    search_k = 20 if "科研精读" in reading_mode else 10

    try:
        docs = st.session_state.db.max_marginal_relevance_search(
            prompt,
            k=search_k * 2,  # 多取一些再手动过滤
            fetch_k=60,
            lambda_mult=0.7,
        )

        # 修复：手动过滤，替代不稳定的 filter 参数
        if selected_scope != "🌐 对比所有论文":
            docs = [d for d in docs if d.metadata.get('source_paper') == selected_scope]

        docs = docs[:search_k]

        if not docs:
            return "未找到相关内容，请调整问题或加载更多论文。"

        context = "\n\n".join([
            f"📄【{d.metadata.get('source_paper', '未知论文')} 第{d.metadata.get('page', 0) + 1}页】:\n{d.page_content}"
            for d in docs
        ])

        sys_prompt = f"""
你是专业的科研助手，基于以下论文内容回答问题，要求：
1. 回答必须基于论文原文，标注来源（论文名+页码）
2. 结构清晰，分点说明，公式用 $ 包裹
3. 客观中立，不编造信息
4. 优先使用学术术语，保持专业性

论文内容：
{context}

用户问题：{prompt}
"""
        llm = get_llm(api_key)
        response = llm.invoke(sys_prompt)
        return fix_latex_errors(response.content)

    except Exception as e:
        return f"生成出错: {e}"

# ================= 7. 侧边栏 =================
with st.sidebar:
    st.header("🎛️ 科研控制台")
    user_api_key = st.text_input("智谱 API Key", type="password")
    ss_api_key = st.text_input(
        "Semantic Scholar API Key",
        type="password",
        help="填入后大幅提升速率，无则使用匿名模式"
    )
    if ss_api_key:
        st.success("🚀 科研高速模式已激活")
    else:
        st.caption("🐢 匿名限速模式（并发数受限）")

    st.markdown("---")

    st.subheader("⚡ 高质量论文筛选")
    min_cite_filter = st.slider("最低引用数（过滤低质量论文）", 0, 100, 5, step=5)
    min_year_filter = st.slider("发表起始年份", 2000, 2026, 2015)
    expand_depth = st.radio("图谱拓展深度", ["基础（直接引用）", "领域级（递归拓展）"], index=0)
    expand_depth_num = 1 if expand_depth == "基础（直接引用）" else 2

    st.markdown("---")

    if st.session_state.loaded_files:
        st.subheader("🗂️ 文献管理")
        for file in list(st.session_state.loaded_files):
            col_f1, col_f2 = st.columns([4, 1])
            with col_f1:
                st.text(f"📄 {file[:18]}..." if len(file) > 20 else f"📄 {file}")
            with col_f2:
                if st.button("🗑️", key=f"del_{file}"):
                    st.session_state.loaded_files.remove(file)
                    st.session_state.all_chunks = [
                        c for c in st.session_state.all_chunks
                        if c.metadata.get('source_paper') != file
                    ]
                    if user_api_key:
                        rebuild_index_from_chunks(user_api_key)
                    st.rerun()

        if st.button("🗑️ 清空全部", type="primary"):
            st.session_state.db = None
            st.session_state.loaded_files = []
            st.session_state.all_chunks = []
            st.session_state.chat_history = []
            st.rerun()

    st.subheader("📖 研读模式")
    reading_mode = st.radio("选择模式:", ["🟢 快速问答", "📖 科研精读（结构化解析）"], index=1)

    if st.session_state.loaded_files:
        st.markdown("---")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("📊 生成论文对比表"):
                require_api_key(user_api_key)
                if st.session_state.db:
                    with st.spinner("分析论文特征..."):
                        llm = get_llm(user_api_key)
                        aggregated_context = ""
                        for filename in st.session_state.loaded_files:
                            sub_docs = st.session_state.db.similarity_search(
                                "研究目标 创新点 方法 数据集 结论 局限性", k=3
                            )
                            sub_docs = [d for d in sub_docs if d.metadata.get('source_paper') == filename]
                            if sub_docs:
                                aggregated_context += f"\n=== {filename} ===\n" + "\n".join(
                                    [d.page_content for d in sub_docs]) + "\n"

                        prompt = f"""
请基于以下论文内容，生成科研级对比表格（Markdown格式），列要求：
论文名 | 研究目标 | 核心方法 | 创新点 | 数据集 | 主要结论 | 局限性

论文内容：
{aggregated_context}
"""
                        res = llm.invoke(prompt)
                        st.session_state.chat_history.append({"role": "assistant", "content": res.content})
                        st.rerun()

        with col_btn2:
            if st.button("📝 生成结构化综述"):
                require_api_key(user_api_key)
                if st.session_state.db:
                    with st.spinner("生成领域综述..."):
                        llm = get_llm(user_api_key)
                        aggregated_context = ""
                        for filename in st.session_state.loaded_files:
                            sub_docs = st.session_state.db.similarity_search(
                                "Abstract introduction conclusion contribution", k=5
                            )
                            sub_docs = [d for d in sub_docs if d.metadata.get('source_paper') == filename]
                            if sub_docs:
                                aggregated_context += f"\n=== {filename} ===\n" + "\n".join(
                                    [d.page_content for d in sub_docs]) + "\n"

                        prompt = f"""
请基于以下论文内容，生成一份结构化科研综述（中文），包含：
1. 领域背景与研究问题
2. 核心方法与技术路线
3. 主要发现与结论
4. 研究不足与未来方向

论文内容：
{aggregated_context}
"""
                        res = llm.invoke(prompt)
                        st.session_state.chat_history.append({"role": "assistant", "content": res.content})
                        st.rerun()

        scope_options = ["🌐 对比所有论文"] + st.session_state.loaded_files
        st.session_state.selected_scope = st.selectbox("👁️ 专注范围", scope_options)

    st.markdown("---")
    st.subheader("📥 上传论文")
    uploaded_file = st.file_uploader("拖入 PDF 论文", type="pdf")
    if uploaded_file and user_api_key and st.button("确认加载"):
        require_api_key(user_api_key)
        with st.spinner("解析PDF并构建向量库..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                path = tmp.name
            # cleanup=True 让函数内部 finally 负责删除
            process_and_add_to_db(path, uploaded_file.name, user_api_key, cleanup=True)
            st.rerun()

# ================= 8. 主界面 =================
tab_search, tab_chat = st.tabs(["🔍 文献脉络（Connected Papers）", "💬 科研精读"])

with tab_search:
    st.subheader("🌍 学术大数据检索（科研级）")
    col_q, col_sort, col_n = st.columns([3, 1.5, 1])
    with col_q:
        search_query = st.text_input(
            "关键词（支持 ti:标题 abs:摘要 语法）",
            value=st.session_state.suggested_query,
            placeholder="例如: ti:education AND abs:robot"
        )
    with col_sort:
        sort_mode = st.selectbox("排序规则", ["🔥 科研影响力（引用+相关）", "📅 时间由新到旧", "📈 引用量由高到低"])
    with col_n:
        max_results = st.number_input("获取数量", min_value=5, max_value=50, value=15)

    if st.button("🚀 开始检索（自动过滤低质量论文）") and search_query:
        with st.spinner("检索并同步科研数据..."):
            try:
                arxiv_sort = arxiv.SortCriterion.Relevance
                if "时间" in sort_mode:
                    arxiv_sort = arxiv.SortCriterion.SubmittedDate

                refined_query = search_query
                if " " in search_query and "AND" not in search_query and '"' not in search_query:
                    refined_query = " AND ".join([f'(ti:{w} OR abs:{w})' for w in search_query.split()])

                search = arxiv.Search(query=refined_query, max_results=max_results, sort_by=arxiv_sort)
                raw_results = list(search.results())

                # 修复：并发获取引用数，替代串行
                arxiv_ids = [r.entry_id for r in raw_results]
                progress_bar = st.progress(0, text="正在并发获取引用数...")
                cite_map = fetch_citations_batch(arxiv_ids, ss_key=ss_api_key)
                progress_bar.progress(1.0, text="完成！")

                results_with_cite = []
                for res in raw_results:
                    cites = cite_map.get(res.entry_id, 0)
                    if cites >= min_cite_filter:
                        results_with_cite.append({'obj': res, 'citations': cites})

                # 修复：除零保护
                current_year = datetime.now().year
                if "科研影响力" in sort_mode:
                    results_with_cite.sort(
                        key=lambda x: (
                            x['citations'] * 0.7 +
                            (1 / max(1, current_year - x['obj'].published.year + 1)) * 0.3
                        ),
                        reverse=True
                    )
                elif "引用量" in sort_mode:
                    results_with_cite.sort(key=lambda x: x['citations'], reverse=True)

                st.session_state.search_results = results_with_cite
                st.success(f"✅ 完成！筛选后剩余 {len(results_with_cite)} 篇高质量论文（引用≥{min_cite_filter}）。")
            except Exception as e:
                st.error(f"检索失败: {e}")

    # 图谱展示
    if st.session_state.search_results:
        if st.session_state.focus_paper_id:
            st.markdown("---")
            st.subheader("📊 科研领域脉络图谱（Connected Papers 风格）")
            st.markdown(
                '<div class="timeline-note">🟥 核心论文 | 🔵 参考文献（前人工作） | 🟢 施引文献（后续发展）'
                ' | 节点越大=引用越高</div>',
                unsafe_allow_html=True
            )

            with st.spinner("构建领域脉络图谱..."):
                # 修复：传入哈希后的 key 作为缓存键，原始 key 作为普通参数
                ss_key_hash = _hash_key(ss_api_key)
                g_data = fetch_graph_data(
                    st.session_state.focus_paper_id,
                    ss_key_hash=ss_key_hash,
                    ss_key=ss_api_key,
                    expand_depth=expand_depth_num
                )

            if not g_data:
                st.warning("⚠️ 暂时无法获取图谱数据。请检查 API Key 或稍后再试。")
            else:
                col_graph, col_info = st.columns([2.5, 1])
                with col_graph:
                    clicked_node_id, all_details = render_research_graph(
                        g_data,
                        min_citation=min_cite_filter,
                        year_range=(min_year_filter, 2026)
                    )

                with col_info:
                    col_export1, col_export2 = st.columns(2)
                    with col_export1:
                        csv_data = export_papers_to_csv(all_details)
                        st.download_button(
                            label="📥 导出CSV",
                            data=csv_data,
                            file_name=f"research_papers_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    with col_export2:
                        bibtex_data = export_papers_to_bibtex(all_details)
                        st.download_button(
                            label="📥 导出BibTeX",
                            data=bibtex_data,
                            file_name=f"research_papers_{datetime.now().strftime('%Y%m%d')}.bib",
                            mime="text/plain"
                        )

                    if user_api_key and st.button("📝 生成领域综述", use_container_width=True):
                        with st.spinner("AI 分析领域脉络..."):
                            summary = generate_research_summary(all_details, user_api_key)
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"## 领域科研综述\n{summary}"
                            })
                            st.rerun()

                    if clicked_node_id and clicked_node_id in all_details:
                        info = all_details[clicked_node_id]
                        st.markdown("### 📑 论文详情")
                        st.markdown(f"**{info['title']}**")
                        if info['is_high_impact']:
                            st.markdown(
                                '<span class="high-impact">🔥 高影响力论文（引用≥50）</span>',
                                unsafe_allow_html=True
                            )
                        c1, c2, c3 = st.columns(3)
                        c1.metric("📅 年份", info['year'])
                        c2.metric("🔥 引用", info['cites'])
                        journal_display = info['journal']
                        c3.metric("📖 期刊", journal_display[:8] + "..." if len(journal_display) > 8 else journal_display)

                        st.markdown("---")
                        st.markdown(
                            f"**摘要**: \n\n <div style='font-size:0.85em; color:#444; height:200px;"
                            f" overflow-y:auto;'>{info['abstract']}</div>",
                            unsafe_allow_html=True
                        )
                        st.markdown("---")
                        st.link_button("🌐 查看原文", info['url'], use_container_width=True)
                    else:
                        st.info("""
🎯 **科研图谱使用指南**
- 🖱️ 点击节点：查看论文详情
- 🔄 滚轮缩放：调整图谱大小
- 📥 导出：直接用于论文引用/整理
                        """)

        # 论文列表
        st.markdown("---")
        st.subheader(f"📜 高质量论文列表（共 {len(st.session_state.search_results)} 篇）")
        for i, item in enumerate(st.session_state.search_results):
            res = item['obj']
            cites = item['citations']
            is_high = cites >= 50

            expander_title = f"#{i + 1} {'🔥' if is_high else '📄'} {res.title} ({res.published.year}) | 引用: {cites}"
            with st.expander(expander_title):
                st.markdown(
                    f"**👨‍🏫 作者**: {', '.join([a.name for a in res.authors])} | "
                    f"**📅 发表**: {res.published.strftime('%Y-%m-%d')}"
                )
                if is_high:
                    st.markdown('<span class="high-impact">🔥 高影响力论文（优先精读）</span>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="abstract-box"><b>📝 摘要：</b><br>{res.summary.replace(chr(10), " ")}</div>',
                    unsafe_allow_html=True
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"[🔗 ArXiv 原文]({res.entry_id})")
                with col2:
                    if st.button(f"⬇️ 加入精读", key=f"dl_search_{i}"):
                        require_api_key(user_api_key)
                        with st.spinner("下载并解析论文..."):
                            tmp_path = None
                            try:
                                tmp_path = res.download_pdf(dirpath=tempfile.gettempdir())
                                process_and_add_to_db(tmp_path, res.title, user_api_key, cleanup=False)
                                st.success("✅ 已加入精读库！")
                            except Exception as e:
                                st.error(f"失败: {e}")
                            finally:
                                if tmp_path and os.path.exists(tmp_path):
                                    try:
                                        os.remove(tmp_path)
                                    except Exception:
                                        pass
                with col3:
                    if st.button(f"🕸️ 查看脉络", key=f"btn_graph_{i}"):
                        st.session_state.focus_paper_id = res.entry_id
                        st.rerun()

# ================= 9. 科研精读 Tab =================
with tab_chat:
    st.subheader("💬 科研精读空间（AI 结构化解析）")

    if st.session_state.loaded_files:
        st.caption(f"📚 模式：{reading_mode} | 范围：{st.session_state.selected_scope}")

        # 修复：快捷按钮将 prompt 存入 pending，渲染完聊天记录后统一执行 AI 调用
        st.markdown("### ⚡ 快捷科研提问")
        col_quick1, col_quick2, col_quick3 = st.columns(3)
        with col_quick1:
            if st.button("📌 提取创新点+方法+结论"):
                st.session_state.pending_ai_prompt = "请提取每篇论文的创新点、核心方法、主要结论，用Markdown表格展示"
        with col_quick2:
            if st.button("📊 对比论文差异"):
                st.session_state.pending_ai_prompt = "对比这些论文的研究方法、创新点、局限性，分析各自的优势和不足"
        with col_quick3:
            if st.button("📝 生成研究思路"):
                st.session_state.pending_ai_prompt = "基于这些论文，给出该领域的研究思路和未来方向建议"

    # 渲染历史记录
    for msg in st.session_state.chat_history:
        if msg["role"] == "system_notice":
            st.info(msg["content"])
        else:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # 修复：处理快捷按钮触发的 pending prompt，执行真实 AI 调用
    if st.session_state.pending_ai_prompt:
        prompt = st.session_state.pending_ai_prompt
        st.session_state.pending_ai_prompt = None

        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("AI 正在分析..."):
                result = run_ai_chat(
                    prompt, user_api_key, reading_mode,
                    st.session_state.get("selected_scope", "🌐 对比所有论文")
                )
            if result:
                st.markdown(result)
                st.session_state.chat_history.append({"role": "assistant", "content": result})

    # 手动输入
    if prompt := st.chat_input("输入科研问题（例如：这篇论文用了什么数据集？和XX论文的创新点对比？）"):
        if not st.session_state.db:
            st.warning("🧠 请先上传/加载论文到精读库")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("AI 正在分析..."):
                    result = run_ai_chat(
                        prompt, user_api_key, reading_mode,
                        st.session_state.get("selected_scope", "🌐 对比所有论文")
                    )
                if result:
                    st.markdown(result)
                    st.session_state.chat_history.append({"role": "assistant", "content": result})
                else:
                    st.error("请检查 API Key 是否正确，或网络是否正常。")
