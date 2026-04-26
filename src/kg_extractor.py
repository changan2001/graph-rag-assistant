"""
知识图谱抽取模块（高并发批处理版）

功能：
    利用大语言模型从文本切片中抽取实体和关系，
    采用“合并批处理 + 轻量级多线程”策略，
    大幅降低 API 调用频率，突破抽取速度瓶颈。
"""

import json
import time
import concurrent.futures
from src.llm_client import LLMClient


class KGExtractor:
    """
    知识图谱抽取器
    """

    def __init__(self, llm_client: LLMClient, batch_size: int = 10, max_workers: int = 3):
        """
        初始化抽取器

        参数:
            llm_client: 已经实例化好的大语言模型客户端
            batch_size: 每个批次合并的文本块数量（默认 10 块，约 5000 字符，黄金平衡点）
            max_workers: 最大并发线程数（默认 3 个，避免触发高频 API 限流）
        """
        self.llm_client = llm_client
        self.batch_size = batch_size
        self.max_workers = max_workers

    def _extract_from_text(self, text: str, max_retries: int = 2) -> list[dict]:
        """
        内部方法：从一段合并后的长文本中抽取三元组，包含限流重试机制
        """
        system_prompt = """你是一个专业的知识图谱抽取专家。
请从用户提供的长文本中抽取核心实体和关系，并严格按照以下JSON格式输出。
不要输出任何解释说明，只输出合法的JSON格式内容。

要求：
1. 实体类型（label）建议为：概念、技术、框架、模块、函数、属性、工具、组件、人物、地点等。
2. 关系类型建议为：包含、依赖、实现、用于、替代、属于、关联、支持、发明等。
3. 尽可能多地抽取有意义的核心关系，忽略无意义的过渡句。

输出格式：
{
    "triplets": [
        {
            "head": {"name": "实体A", "label": "类型"},
            "relation": "关系",
            "tail": {"name": "实体B", "label": "类型"}
        }
    ]
}"""

        user_message = f"请抽取以下文本中的核心实体和相互关系：\n{text}"

        for attempt in range(max_retries):
            # chat_with_json_output 内部已经包含了完整的 JSON 清洗逻辑
            response_text = self.llm_client.chat_with_json_output(user_message, system_prompt)

            try:
                data = json.loads(response_text)
                triplets = data.get("triplets", [])
                return triplets
            except json.JSONDecodeError:
                if attempt < max_retries - 1:
                    print(f"  ⚠️ JSON解析失败或被限流，正在冷静 2 秒后重试 (第 {attempt + 1} 次重试)...")
                    time.sleep(2)  # 指数退避：冷静 2 秒后重试
                    continue
                print("  ❌ 多次尝试解析 JSON 均失败，跳过此批次。")
                return []

        return []

    def extract_triplets_from_chunks(self, chunks: list[dict], progress_callback=None) -> list[dict]:
        """
        核心对外接口：接收所有的文本块，执行批处理与多线程并发抽取

        参数:
            chunks: 所有文档切片的列表
            progress_callback: 进度回调函数，用于更新前端进度条

        返回:
            所有抽取出的大量三元组列表
        """
        if not chunks:
            return []

        # 1. 化零为整：将切片按 batch_size 打包
        batches = []
        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i:i + self.batch_size]
            # 用换行符将多个文本块拼接成一段长文本
            combined_text = "\n\n".join([c["content"] for c in batch_chunks])
            batches.append(combined_text)

        print(f"📦 共 {len(chunks)} 个文本块，已合并为 {len(batches)} 个批次准备抽取。")

        all_triplets = []
        completed_batches = 0

        # 2. 多管齐下：使用线程池并发处理批次
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有批次任务给线程池
            future_to_batch = {executor.submit(self._extract_from_text, text): text for text in batches}

            # as_completed 会在某个线程完成时立即 yield 返回
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    triplets = future.result()
                    all_triplets.extend(triplets)
                except Exception as e:
                    print(f"批处理抽取时发生系统错误: {e}")

                # 更新进度
                completed_batches += 1
                if progress_callback:
                    progress_callback(completed_batches, len(batches))

        print(f"✅ 所有批次抽取完毕，共获得 {len(all_triplets)} 个三元组。")
        return all_triplets


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    import config

    print("🚀 开始测试大模型图谱 高并发批处理 抽取能力...")

    if not config.LLM_API_KEY:
        print("❌ 请先在 .env 文件中配置 LLM_API_KEY")
    else:
        client = LLMClient(
            api_base=config.LLM_API_BASE,
            api_key=config.LLM_API_KEY,
            model_name=config.LLM_MODEL,
        )

        extractor = KGExtractor(client, batch_size=2, max_workers=2)

        # 模拟 3 个切片
        test_chunks = [
            {"content": "RAG是一种结合信息检索和文本生成的技术。"},
            {"content": "LangChain框架完美支持了RAG技术的落地。"},
            {"content": "知识图谱可以与RAG结合形成GraphRAG。"}
        ]

        print(f"🧠 大模型正在使用批处理并发抽取，请稍候...\n")

        triplets = extractor.extract_triplets_from_chunks(test_chunks)

        print("\n✅ 抽取完成！结果如下：")
        print(json.dumps(triplets, ensure_ascii=False, indent=2))