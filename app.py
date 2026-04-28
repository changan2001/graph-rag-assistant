"""
GraphRAG 智能文档助手 - Streamlit 前端交互界面

功能：
    1. 提供可视化的 PDF 上传和处理界面
    2. 提供知识图谱的交互式可视化
    3. 提供融合向量与图谱检索的对话界面
"""

import os
import tempfile
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

import config
from src.document_loader import load_and_process_pdf
from src.llm_client import LLMClient
from src.vector_store import VectorStore
from src.kg_extractor import KGExtractor
from src.kg_store import KGStore
from src.qa_chain import GraphRAGChain

# ============================================================
# 页面基础设置
# ============================================================
st.set_page_config(
    page_title="GraphRAG 智能文档助手",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 全局资源初始化（使用缓存避免重复加载）
# ============================================================
@st.cache_resource
def init_components():
    """初始化所有的核心组件并放入缓存"""
    llm = LLMClient(config.LLM_API_BASE, config.LLM_API_KEY, config.LLM_MODEL)
    vs = VectorStore(config.EMBED_API_BASE, config.EMBED_API_KEY, config.EMBED_MODEL)
    kg = KGStore()
    extractor = KGExtractor(llm, batch_size=10, max_workers=3) # 启用高并发批处理配置
    chain = GraphRAGChain(llm, vs, kg)
    return llm, vs, kg, extractor, chain

# 检查密钥是否配置
if not config.LLM_API_KEY or not config.EMBED_API_KEY or not config.NEO4J_PASSWORD:
    st.error("❌ 错误：请先在项目根目录的 .env 文件中配置好所有的 API 密钥！")
    st.stop()

# 获取初始化的组件
llm_client, vector_store, kg_store, kg_extractor, qa_chain = init_components()

# 固定的集合名称，用于存放用户上传的文档
COLLECTION_NAME = "uploaded_documents"

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.title("🧠 GraphRAG 系统状态")

    # 显示数据库统计信息
    st.markdown("### 📊 数据库统计")
    try:
        doc_count = vector_store.get_collection_count(COLLECTION_NAME)
        st.metric("向量库文档块数量", doc_count)
    except Exception:
        st.metric("向量库文档块数量", 0)

    try:
        nodes = kg_store.get_all_nodes()
        edges = kg_store.get_all_edges()
        col1, col2 = st.columns(2)
        col1.metric("图谱节点数", len(nodes))
        col2.metric("图谱关系数", len(edges))
    except Exception:
        st.metric("图谱状态", "未连接或为空")

    st.markdown("---")
    st.markdown("### ⚠️ 危险操作")
    if st.button("🗑️ 清空所有数据", type="primary"):
        with st.spinner("正在清空数据..."):
            vector_store.delete_collection(COLLECTION_NAME)
            kg_store.clear_database()
            # 清理历史对话
            if "messages" in st.session_state:
                st.session_state.messages = []
        st.success("数据已全部清空！请刷新页面。")
        st.rerun()

# ============================================================
# 主界面布局
# ============================================================
st.title("📚 GraphRAG 智能文档助手")
st.markdown("融合 **向量检索 (FAISS)** 与 **知识图谱 (Neo4j)** 的新一代文档对话助手。")

# 创建三个选项卡
tab_doc, tab_graph, tab_chat = st.tabs(["📄 文档管理", "🕸️ 知识图谱视图", "💬 智能问答"])

# ------------------------------------------------------------
# 选项卡 1：文档管理
# ------------------------------------------------------------
with tab_doc:
    st.header("上传并处理文档")
    st.info("上传的 PDF 将被切片，文本存入向量库，同时从文本中抽取三元组存入知识图谱。")

    uploaded_file = st.file_uploader("请选择一个 PDF 文件", type=["pdf"])

    if uploaded_file is not None:
        if st.button("🚀 开始处理文档", type="primary"):
            # 使用临时文件保存上传的PDF，以便 pdfplumber 读取
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_pdf_path = tmp_file.name

            try:
                # 步骤 1：解析和切片
                with st.status("正在解析文档...", expanded=True) as status:
                    st.write("1. 正在提取文本并切片...")
                    chunks = load_and_process_pdf(tmp_pdf_path)
                    st.write(f"✅ 完成！共切分为 {len(chunks)} 个文本块。")

                    # 步骤 2：写入向量库
                    st.write("2. 正在进行向量化并存入 FAISS...")
                    vector_store.add_chunks(chunks, collection_name=COLLECTION_NAME)
                    st.write("✅ 向量库写入完成！")

                    # 步骤 3：抽取知识图谱（采用全新的多线程批处理机制）
                    st.write("3. 正在抽取知识图谱数据...")
                    progress_bar = st.progress(0)

                    # 定义回调函数，用于在多线程抽取时实时更新进度条
                    def update_progress(completed_batches, total_batches):
                        progress_bar.progress(completed_batches / total_batches)

                    # 一键呼叫底层高并发接口
                    all_triplets = kg_extractor.extract_triplets_from_chunks(
                        chunks,
                        progress_callback=update_progress
                    )

                    st.write(f"✅ 抽取完成！共发现 {len(all_triplets)} 个潜在关系。")

                    # 步骤 4：写入 Neo4j
                    st.write("4. 正在安全写入 Neo4j 图数据库...")
                    kg_store.add_triplets(all_triplets)
                    st.write("✅ 图谱写入完成！")

                    status.update(label="文档处理完毕！", state="complete", expanded=False)

                st.balloons()
                st.success("文档处理完成！可前往【知识图谱视图】查看结果。")

            except Exception as e:
                st.error(f"处理文档时发生错误: {e}")
            finally:
                # 清理临时文件
                if os.path.exists(tmp_pdf_path):
                    os.remove(tmp_pdf_path)

# ------------------------------------------------------------
# 选项卡 2：知识图谱视图
# ------------------------------------------------------------
with tab_graph:
    st.header("知识图谱可视化")

    if st.button("🔄 刷新图谱"):
        st.rerun()

    try:
        nodes_data = kg_store.get_all_nodes()
        edges_data = kg_store.get_all_edges()

        if not nodes_data:
            st.warning("当前图谱为空，请先在【文档管理】中上传并处理文档。")
        else:
            # 将 Neo4j 数据转换为 agraph 可识别的格式
            nodes = []
            edges = []

            # 记录已添加的节点，防止重复
            added_nodes = set()

            for node in nodes_data:
                node_name = node["name"]
                if node_name not in added_nodes:
                    nodes.append(Node(
                        id=node_name,
                        label=node_name,
                        size=25,
                        shape="dot"
                    ))
                    added_nodes.add(node_name)

            for edge in edges_data:
                # 确保边的两端节点都存在
                if edge["source"] in added_nodes and edge["target"] in added_nodes:
                    edges.append(Edge(
                        source=edge["source"],
                        label=edge["relation"],
                        target=edge["target"]
                    ))

            # 图谱可视化配置
            config = Config(
                width="100%",
                height=600,
                directed=True,
                physics=True,
                hierarchical=False,
                nodeHighlightBehavior=True,
                highlightColor="#F7A7A6",
                collapsible=True
            )

            # 渲染图谱
            return_value = agraph(nodes=nodes, edges=edges, config=config)

    except Exception as e:
        st.error(f"渲染图谱时出错: {e}")

# ------------------------------------------------------------
# 选项卡 3：智能问答
# ------------------------------------------------------------
with tab_chat:
    st.header("多模态知识交互")

    # 初始化会话状态中的聊天历史
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "你好！我是你的 GraphRAG 助手。我已经准备好解答关于上传文档的任何问题了。"}
        ]

    # 展示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # 如果历史消息中有检索上下文，用折叠面板展示
            if "context" in msg:
                with st.expander("🔍 查看大模型检索过程"):
                    st.json(msg["context"])

    # 聊天输入框
    if prompt := st.chat_input("你想问什么？"):
        # 1. 立即显示用户的问题
        with st.chat_message("user"):
            st.markdown(prompt)
        # 记录用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 2. 生成助手回答
        with st.chat_message("assistant"):
            with st.spinner("正在执行混合检索..."):
                # 调用 GraphRAG 核心链
                result = qa_chain.ask(prompt, collection_name=COLLECTION_NAME)
                answer = result["answer"]

                # 提取检索上下文，用于展示
                context_data = {
                    "提取到的实体": result["entities_extracted"],
                    "用到的图谱路径": result["graph_context"].splitlines() if result["graph_context"] else ["无"],
                    "检索到的文本片段数": len(result["vector_context"])
                }

                # 渲染回答内容
                st.markdown(answer)
                # 渲染检索过程
                with st.expander("🔍 查看检索与推理过程"):
                    st.json(context_data)
                    st.markdown("**相关文档原文示例：**")
                    for i, doc in enumerate(result["vector_context"][:2]): # 只展示前2个避免太长
                        st.text(f"片段 {i+1}: {doc['content']}")

                # 记录助手消息
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "context": context_data
                })
