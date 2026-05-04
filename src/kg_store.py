"""
Neo4j 知识图谱存储模块（最终安全版）

功能：
    1. 连接云端 Neo4j Aura 图数据库
    2. 将抽取出的三元组数据安全写入图数据库
    3. 增加写入前的数据结构二次校验，防止脏数据注入
"""

import re
from neo4j import GraphDatabase
import config


def sanitize_relation_type(relation: str) -> str:
    """
    清理关系类型名称，确保它可以安全用作Cypher关系类型
    """
    if not relation or not isinstance(relation, str):
        return "关联"
    # 只保留中文字符、英文字母、数字和下划线
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]", "_", relation)
    # 去掉首尾的下划线
    cleaned = cleaned.strip("_")
    return cleaned if cleaned else "关联"


class KGStore:
    """
    知识图谱数据库交互类
    """

    def __init__(self):
        """初始化数据库连接"""
        try:
            self.driver = GraphDatabase.driver(
                config.NEO4J_URI,
                auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
            )
            self.driver.verify_connectivity()
            print("成功连接到 Neo4j Aura 云端数据库！")
        except Exception as e:
            print(f"连接 Neo4j 数据库失败: {e}")
            self.driver = None

    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()

    def clear_database(self):
        """清空整个数据库（谨慎使用）"""
        if not self.driver:
            return

        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("已清空 Neo4j 数据库中的所有数据。")

    def add_triplets(self, triplets: list[dict]):
        """
        将三元组列表写入图数据库（带严格类型校验）
        """
        if not self.driver or not triplets:
            return

        success_count = 0
        with self.driver.session() as session:
            for triplet in triplets:
                # 防御性编程：确保传入的 triplet 是字典
                if not isinstance(triplet, dict):
                    continue

                head = triplet.get("head", {})
                tail = triplet.get("tail", {})
                relation = triplet.get("relation", "关联")

                # 防御性编程：确保 head 和 tail 也是字典
                if not isinstance(head, dict) or not isinstance(tail, dict):
                    continue

                head_name = str(head.get("name", "")).strip()
                head_label = str(head.get("label", "Entity")).strip()
                tail_name = str(tail.get("name", "")).strip()
                tail_label = str(tail.get("label", "Entity")).strip()

                if not head_name or not tail_name:
                    continue

                # 清理标签和关系类型中的特殊字符
                head_label = sanitize_relation_type(head_label)
                tail_label = sanitize_relation_type(tail_label)
                safe_relation = sanitize_relation_type(str(relation))

                # 使用反引号包裹动态标签和关系类型，确保Cypher语法安全
                cypher = f"""
                MERGE (h:`{head_label}` {{name: $head_name}})
                MERGE (t:`{tail_label}` {{name: $tail_name}})
                MERGE (h)-[r:`{safe_relation}`]->(t)
                """

                try:
                    session.run(cypher, head_name=head_name, tail_name=tail_name)
                    success_count += 1
                except Exception as e:
                    print(f"  写入失败: [{head_name}]-({relation})->[{tail_name}] 错误: {e}")

        print(f"成功将处理好的 {success_count} 个关系写入 Neo4j 数据库。")

    def query_by_entity(self, entity_name: str, max_relations: int = None) -> list[dict]:
        """根据实体名称查询相关的关系"""
        if not self.driver or not entity_name:
            return []

        if max_relations is None:
            max_relations = config.KG_MAX_RELATIONS

        results_list = []
        with self.driver.session() as session:
            cypher = """
            MATCH (n)-[r]-(m)
            WHERE n.name CONTAINS $entity
            RETURN n.name AS source, type(r) AS relation, m.name AS target
            LIMIT $limit
            """
            try:
                results = session.run(cypher, entity=entity_name, limit=max_relations)
                for record in results:
                    results_list.append({
                        "source": record["source"],
                        "relation": record["relation"],
                        "target": record["target"],
                    })
            except Exception as e:
                print(f"图谱查询出错: {e}")

        return results_list

    def get_all_nodes(self) -> list[dict]:
        """获取图谱中的所有节点（用于可视化）"""
        if not self.driver:
            return []

        nodes = []
        with self.driver.session() as session:
            try:
                results = session.run("MATCH (n) RETURN n.name AS name, labels(n) AS labels")
                for record in results:
                    nodes.append({
                        "name": record["name"],
                        "label": record["labels"] if record["labels"] else "Entity",
                    })
            except Exception as e:
                print(f"获取节点出错: {e}")

        return nodes

    def get_all_edges(self) -> list[dict]:
        """获取图谱中的所有边/关系（用于可视化）"""
        if not self.driver:
            return []

        edges = []
        with self.driver.session() as session:
            try:
                results = session.run(
                    "MATCH (n)-[r]->(m) RETURN n.name AS source, type(r) AS relation, m.name AS target"
                )
                for record in results:
                    edges.append({
                        "source": record["source"],
                        "relation": record["relation"],
                        "target": record["target"],
                    })
            except Exception as e:
                print(f"获取边出错: {e}")

        return edges


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("开始测试 Neo4j Aura 连接...")

    store = KGStore()

    test_data = [
        {
            "head": {"name": "Python", "label": "编程语言"},
            "relation": "用于",
            "tail": {"name": "大模型开发", "label": "技术领域"},
        }
    ]

    store.add_triplets(test_data)

    # 测试查询
    print("\n--- 查询测试 ---")
    results = store.query_by_entity("Python")
    for r in results:
        print(f"  {r['source']} -[{r['relation']}]-> {r['target']}")

    store.close()
    print("\n测试完毕。")
