"""
向量数据库模块（FAISS 版 - 修复批量限制问题）

功能：
    1. 使用云端大模型 API 生成向量
    2. 使用 FAISS 进行本地高速检索
    3. 通过元数据手动实现集合隔离
"""

import os
import shutil
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from openai import OpenAI
import config


class RemoteEmbeddings(Embeddings):
    """
    通过调用兼容 OpenAI 格式的 Embedding API 来生成向量
    """

    def __init__(self, api_base: str, api_key: str, model_name: str):
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量把文本转成向量（增加分批处理逻辑以绕过 API 限制）"""
        batch_size = 50  # 大多数 API 限制最大 64，这里设为 50 比较安全
        all_embeddings = []

        try:
            # 将所有文本切片分成多个批次（例如 98 个切片分成 50 + 48）
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                response = self.client.embeddings.create(
                    input=batch_texts,
                    model=self.model_name,
                )

                # 将这一批次的向量结果追加到总列表中
                all_embeddings.extend([data.embedding for data in response.data])

            return all_embeddings

        except Exception as e:
            print(f"调用 Embedding API 获取向量失败: {e}")
            raise e

    def embed_query(self, text: str) -> list[float]:
        """把单条提问转成向量"""
        return self.embed_documents([text])[0]


class VectorStore:
    """
    向量数据库管理器 (FAISS版)
    """

    def __init__(self, api_base: str, api_key: str, model_name: str, persist_directory: str = None):
        """
        初始化向量数据库

        参数:
            api_base: Embedding API地址
            api_key: Embedding API密钥
            model_name: Embedding模型名称
            persist_directory: 本地持久化目录
        """
        self.persist_directory = persist_directory or config.CHROMA_PERSIST_DIR
        self.embeddings = RemoteEmbeddings(api_base, api_key, model_name)
        self.vectorstore = None

        # 尝试从本地加载已有数据库
        index_path = os.path.join(self.persist_directory, "index.faiss")
        if os.path.exists(index_path):
            try:
                self.vectorstore = FAISS.load_local(
                    self.persist_directory,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                print("已从本地加载 FAISS 向量库")
            except Exception as e:
                print(f"加载本地已有数据库失败: {e}")

    def add_chunks(self, chunks: list[dict], collection_name: str = "documents"):
        """
        将文本切片存入 FAISS

        参数:
            chunks: 文本切片列表
            collection_name: 集合名称（通过元数据区分不同集合）
        """
        if not chunks:
            return

        texts = [chunk["content"] for chunk in chunks]
        metadatas = [{"chunk_id": chunk["chunk_id"], "collection": collection_name} for chunk in chunks]

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        else:
            self.vectorstore.add_texts(texts, metadatas=metadatas)

        os.makedirs(self.persist_directory, exist_ok=True)
        self.vectorstore.save_local(self.persist_directory)
        print(f"成功将 {len(chunks)} 个文本块存入 FAISS 向量库！")

    def search(self, query: str, collection_name: str = "documents", top_k: int = None) -> list[dict]:
        """
        根据问题检索最相似的文本块

        参数:
            query: 查询文本
            collection_name: 要搜索的集合名称
            top_k: 返回结果数量
        """
        if top_k is None:
            top_k = config.VECTOR_SEARCH_TOP_K

        if self.vectorstore is None:
            print("向量库为空，没有可检索的内容。")
            return []

        # 先检索较多的结果（top_k的3倍），然后手动按collection过滤
        fetch_k = top_k * 3
        try:
            all_results = self.vectorstore.similarity_search_with_score(query, k=fetch_k)
        except Exception as e:
            print(f"向量检索出错: {e}")
            return []

        # 手动过滤：只保留属于指定集合的结果
        search_results = []
        for doc, score in all_results:
            if doc.metadata.get("collection") == collection_name:
                search_results.append({
                    "content": doc.page_content,
                    "chunk_id": doc.metadata.get("chunk_id", -1),
                    "distance": float(score),
                })
                if len(search_results) >= top_k:
                    break

        return search_results

    def delete_collection(self, collection_name: str = "documents"):
        """
        清空数据（删除本地存储文件夹）

        参数:
            collection_name: 集合名称（当前实现为清空全部数据）
        """
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            self.vectorstore = None
            print("已清空本地 FAISS 数据库。")

    def get_collection_count(self, collection_name: str = "documents") -> int:
        """
        获取向量库中的文档总数

        返回:
            文档数量
        """
        if self.vectorstore is None:
            return 0
        try:
            return self.vectorstore.index.ntotal
        except Exception:
            return 0


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("开始测试 FAISS 向量检索引擎...")

    if not config.EMBED_API_KEY:
        print("错误: 请先在 .env 文件中配置 EMBED_API_KEY")
    else:
        vs = VectorStore(
            api_base=config.EMBED_API_BASE,
            api_key=config.EMBED_API_KEY,
            model_name=config.EMBED_MODEL,
        )

        test_chunks = [
            {"content": "LangChain是一个用于开发大模型应用的框架，它提供多种组件。", "chunk_id": 0},
            {"content": "RAG是一种结合检索和生成的技术，让大模型基于外部知识回答问题。", "chunk_id": 1},
            {"content": "知识图谱是用图结构表示知识的方法，节点表示实体，边表示关系。", "chunk_id": 2},
        ]

        vs.add_chunks(test_chunks, collection_name="test_collection")

        print("\n--- 检索测试 ---")
        query = "什么是RAG技术？"
        results = vs.search(query, collection_name="test_collection", top_k=2)

        print(f"查询: {query}")
        for i, result in enumerate(results):
            print(f"结果 {i + 1} (距离: {result['distance']:.4f}): {result['content']}")

        vs.delete_collection("test_collection")
        print("\nFAISS 向量库测试完成！")