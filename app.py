"""
GraphRAG 智能文档助手 - Streamlit 前端交互界面
（流式传输打字机效果 + 智能节流熔断保护版）

功能：
    1. 提供可视化的 PDF 上传和处理界面
    2. 提供知识图谱的交互式可视化
    3. 提供融合向量与图谱检索的流式对话界面（打字机效果）
    4. 内置熔断器状态可视化，出错时一秒定位问题
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
    extractor = KGExtractor(llm, batch_size=10, max_workers=3, circuit_breaker_threshold=3)
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
st.title("📚 基于图谱增强的智能文档问答系统")
st.markdown("融合 **向量检索 (FAISS)** 与 **知识图谱 (Neo4j)** 的 GraphRAG 文档对话助手。")

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
                    st.write("1️⃣ 正在提取文本并切片...")
                    chunks = load_and_process_pdf(tmp_pdf_path)
                    st.write(f"✅ 完成！共切分为 {len(chunks)} 个文本块。")

                    # 步骤 2：写入向量库
                    st.write("2️⃣ 正在进行向量化并存入 FAISS...")
                    vector_store.add_chunks(chunks, collection_name=COLLECTION_NAME)
                    st.write("✅ 向量库写入完成！")

                    # 步骤 3：抽取知识图谱
                    st.write("3️⃣ 正在抽取知识图谱数据...")
                    progress_bar = st.progress(0)

                    # 定义回调函数，用于在多线程抽取时实时更新进度条
                    def update_progress(completed_batches, total_batches):
                        progress_bar.progress(completed_batches / total_batches)

                    # 一键呼叫底层智能节流并发接口
                    all_triplets = kg_extractor.extract_triplets_from_chunks(
                        chunks,
                        progress_callback=update_progress
                    )

                    # ---- 根据熔断器状态和统计数据，给出精确的反馈 ----
                    stats = kg_extractor.last_stats

                    if kg_extractor.is_circuit_broken:
                        # 熔断器触发：红色错误
                        st.error(
                            f"🚨 熔断器触发！API连续 {kg_extractor.circuit_breaker_threshold} 次请求失败！\n\n"
                            f"**可能原因**：API密钥失效、模型名称错误、服务商限流或宕机。\n\n"
                            f"**请检查** `.env` 文件中的 `LLM_API_BASE`、`LLM_API_KEY`、`LLM_MODEL` 配置。\n\n"
                            f"已抢救回 **{len(all_triplets)}** 个三元组（{stats['success_batches']}/{stats['total_batches']} 个批次成功）。"
                        )
                    elif stats["failed_batches"] > 0:
                        # 部分失败：黄色警告
                        st.warning(
                            f"⚠️ 抽取基本完成，但有 {stats['failed_batches']}/{stats['total_batches']} 个批次失败。\n\n"
                            f"成功抽取 **{len(all_triplets)}** 个三元组，图谱可能不完整。"
                        )
                    else:
                        # 全部成功：绿色通过
                        st.write(
                            f"✅ 抽取完成！{stats['success_batches']}/{stats['total_batches']} 个批次全部成功，"
                            f"共发现 **{len(all_triplets)}** 个关系。"
                        )

                    # 步骤 4：写入 Neo4j（即使部分失败，也写入已有数据）
                    if all_triplets:
                        st.write("4️⃣ 正在安全写入 Neo4j 图数据库...")
                        kg_store.add_triplets(all_triplets)
                        st.write("✅ 图谱写入完成！")
                    else:
                        st.write("4️⃣ 没有可写入的三元组，跳过图谱写入。")

                    # 根据最终状态设置进度条的完成状态
                    if kg_extractor.is_circuit_broken:
                        status.update(label="⚠️ 文档处理完成（部分数据因API故障丢失）", state="error", expanded=True)
                    elif stats["failed_batches"] > 0:
                        status.update(label="⚠️ 文档处理完成（有少量批次失败）", state="complete", expanded=False)
                    else:
                        status.update(label="✅ 文档处理完毕！", state="complete", expanded=False)
                        st.balloons()

                st.success("文档处理完成！可前往【知识图谱视图】查看结果，或在【智能问答】中提问。")

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
            agraph_nodes = []
            agraph_edges = []

            # 记录已添加的节点，防止重复
            added_nodes = set()

            for node in nodes_data:
                node_name = node["name"]
                if node_name not in added_nodes:
                    agraph_nodes.append(Node(
                        id=node_name,
                        label=node_name,
                        size=25,
                        shape="dot"
                    ))
                    added_nodes.add(node_name)

            for edge in edges_data:
                # 确保边的两端节点都存在
                if edge["source"] in added_nodes and edge["target"] in added_nodes:
                    agraph_edges.append(Edge(
                        source=edge["source"],
                        label=edge["relation"],
                        target=edge["target"]
                    ))

            # 图谱可视化配置
            graph_config = Config(
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
            return_value = agraph(nodes=agraph_nodes, edges=agraph_edges, config=graph_config)

    except Exception as e:
        st.error(f"渲染图谱时出错: {e}")

# ------------------------------------------------------------
# 选项卡 3：智能问答（流式传输打字机效果）
# ------------------------------------------------------------
with tab_chat:
    st.header("多模态知识交互")

    # 初始化会话状态中的聊天历史
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "你好！我是你的 GraphRAG 助手。我已经准备好解答关于上传文档的问题了。"}
        ]

    # 展示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # 如果历史消息中有检索上下文，用折叠面板展示
            if "context" in msg:
                with st.expander("🔍 查看检索过程"):
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
            # 第一阶段：检索（同步），用 spinner 提示用户"正在检索"
            with st.spinner("🔍 正在检索向量库与知识图谱..."):
                stream, context_data = qa_chain.ask_stream(prompt, collection_name=COLLECTION_NAME)

            # 第二阶段：LLM回答（流式），打字机效果逐字输出
            # st.write_stream 会自动迭代生成器，逐token渲染到界面上
            # 并且在流式结束后，返回拼接好的完整回答文本
            answer = st.write_stream(stream)

            # 组装检索过程数据（供折叠面板展示）
            display_context = {
                "提取到的实体": context_data["entities_extracted"],
                "用到的图谱路径": context_data["graph_context"].splitlines() if context_data["graph_context"] else ["无"],
                "检索到的文本片段数": len(context_data["vector_context"])
            }

            # 渲染检索过程
            with st.expander("🔍 查看推理与检索过程"):
                st.json(display_context)
                st.markdown("**相关文档原文示例：**")
                for i, doc in enumerate(context_data["vector_context"][:2]):
                    st.text(f"片段 {i+1}: {doc['content']}")

            # 记录助手消息到会话历史
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "context": display_context
            })
