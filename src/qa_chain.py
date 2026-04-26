"""
GraphRAG 核心问答链模块（最终版）

功能：
    将向量检索和知识图谱检索融合，实现混合增强的智能问答
"""

import json
from src.llm_client import LLMClient
from src.vector_store import VectorStore
from src.kg_store import KGStore


class GraphRAGChain:
    """
    图谱与向量混合问答链
    """

    def __init__(self, llm_client: LLMClient, vector_store: VectorStore, kg_store: KGStore):
        self.llm = llm_client
        self.vs = vector_store
        self.kg = kg_store

    def _extract_entity_from_query(self, query: str) -> list[str]:
        """从用户问题中提取实体词"""
        system_prompt = """你是一个专业的实体提取器。
请从用户的提问中提取出核心的专有名词、技术概念或实体。
请严格只输出纯JSON格式，不要用```包裹，不要有任何解释文字。
格式要求：{"entities": ["实体1", "实体2"]}
如果没有实体，返回 {"entities": []}"""

        response_text = self.llm.chat_with_json_output(query, system_prompt)

        try:
            data = json.loads(response_text)
            entities = data.get("entities", [])
            print(f"  提取到实体: {entities}")
            return entities
        except json.JSONDecodeError:
            print("  警告: 实体提取JSON解析失败")
            return []

    def _search_graph(self, entities: list[str]) -> str:
        """根据实体去 Neo4j 图谱中查询相关关系"""
        if not entities:
            return ""

        all_relations = []
        for entity in entities:
            relations = self.kg.query_by_entity(entity)
            for r in relations:
                line = f"{r['source']} -[{r['relation']}]-> {r['target']}"
                all_relations.append(line)

        # 去重
        unique_relations = list(set(all_relations))
        return "\n".join(unique_relations)

    def ask(self, query: str, collection_name: str = "documents") -> dict:
        """
        回答用户问题

        参数:
            query: 用户的问题
            collection_name: 向量库中的集合名称

        返回:
            包含回答和检索过程数据的字典
        """
        print(f"\n正在思考问题: {query}")

        # 1. 向量检索
        print("正在从向量库检索相关文档块...")
        vector_results = self.vs.search(query, collection_name=collection_name)
        vector_context = "\n\n".join(
            [f"[文档片段 {i + 1}]: {r['content']}" for i, r in enumerate(vector_results)]
        )
        print(f"  找到 {len(vector_results)} 个相关文档块")

        # 2. 图谱检索
        print("正在从问题中提取实体并检索知识图谱...")
        entities = self._extract_entity_from_query(query)
        graph_context = self._search_graph(entities)
        if graph_context:
            print(f"  找到 {len(graph_context.splitlines())} 条图谱关系")
        else:
            print("  未找到相关图谱关系")

        # 3. 组装终极提示词
        system_prompt = f"""你是一个智能问答助手。
请仔细阅读下方提供的【参考文档片段】和【参考知识图谱关系】，综合这些信息来回答用户的问题。
要求：
1. 回答要逻辑清晰、准确专业。
2. 如果提供的参考信息中没有答案，请诚实地说明"抱歉，参考资料中未提及相关信息"，禁止胡编乱造。
3. 可以在回答中适当提炼和总结。

【参考文档片段】
{vector_context if vector_context else "（无相关文档信息）"}

【参考知识图谱关系】
{graph_context if graph_context else "（无相关图谱信息）"}
"""

        # 4. 生成最终回答
        print("正在生成最终回答...")
        answer = self.llm.chat(query, system_prompt)

        return {
            "answer": answer,
            "vector_context": vector_results,
            "graph_context": graph_context,
            "entities_extracted": entities,
        }


# ============================================================
# 测试代码
# ============================================================
if __name__== "__main__":
    import config

    print("启动 GraphRAG 全链路测试...")

    missing_keys = []
    if not config.LLM_API_KEY:
        missing_keys.append("LLM_API_KEY")
    if not config.EMBED_API_KEY:
        missing_keys.append("EMBED_API_KEY")
    if not config.NEO4J_PASSWORD:
        missing_keys.append("NEO4J_PASSWORD")

    if missing_keys:
        print(f"错误: 请先在 .env 文件中配置: {', '.join(missing_keys)}")
    else:
        llm = LLMClient(config.LLM_API_BASE, config.LLM_API_KEY, config.LLM_MODEL)
        vs = VectorStore(config.EMBED_API_BASE, config.EMBED_API_KEY, config.EMBED_MODEL)
        kg = KGStore()

        # 注入测试知识
        print("\n[准备阶段] 正在向向量库和图谱中注入测试知识...")

        vs.add_chunks(
            [
                {
                    "content": "检索增强生成（RAG）是一种革命性的AI技术。它允许大语言模型在回答问题之前，先去外部的向量数据库里查阅资料，从而避免模型胡说八道（幻觉问题）。RAG的核心流程是：用户提问 - 检索相关文档 - 将文档作为上下文 - LLM生成回答。",
                    "chunk_id": 0,
                },
                {
                    "content": "GraphRAG是RAG技术的进阶版本，它在传统向量检索的基础上，额外引入了知识图谱的结构化信息。通过图谱检索，系统可以发现实体之间的深层关联，从而提供更全面、更准确的回答。",
                    "chunk_id": 1,
                },
            ],
            collection_name="test_collection",
        )

        kg.add_triplets(
            [
                {
                    "head": {"name": "RAG", "label": "技术"},
                    "relation": "依赖",
                    "tail": {"name": "向量数据库", "label": "组件"},
                },
                {
                    "head": {"name": "GraphRAG", "label": "技术"},
                    "relation": "扩展自",
                    "tail": {"name": "RAG", "label": "技术"},
                },
                {
                    "head": {"name": "GraphRAG", "label": "技术"},
                    "relation": "使用",
                    "tail": {"name": "知识图谱", "label": "组件"},
                },
            ]
        )

        # 开始测试问答
        chain = GraphRAGChain(llm, vs, kg)
        question = "什么是GraphRAG？它和普通RAG有什么区别？"

        result = chain.ask(question, collection_name="test_collection")

        print("\n" + "="* 60)
        print(f"提取到的实体词: {result['entities_extracted']}")
        print(f"\n【查询到的图谱关系】:")
        print(result["graph_context"] if result["graph_context"] else "  未找到关系")
        print(f"\n【查询到的文档片段数量】: {len(result['vector_context'])}")
        print(f"\n【最终回答】:\n")
        print(result["answer"])
        print("="* 60)

        # 清理测试数据
        vs.delete_collection("test_collection")
        kg.close()
        print("\n测试数据已清理完毕。")