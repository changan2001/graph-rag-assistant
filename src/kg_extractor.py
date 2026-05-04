"""
知识图谱抽取模块（3线程智能节流 + 熔断保护版）

功能：
    利用大语言模型从文本切片中抽取实体和关系。
    核心并发策略：
    1. 交错发射（Staggered Requests）：每个线程请求前随机延迟，避免同一瞬间撞车
    2. 指数退避重试（Exponential Backoff）：失败后等待时间逐次翻倍，防止重试风暴
    3. 熔断器（Circuit Breaker）：连续多次失败时自动停止，防止静默失败
"""

import json
import time
import random
import threading
import concurrent.futures
from src.llm_client import LLMClient


class KGExtractor:
    """
    知识图谱抽取器（智能节流版）
    """

    def __init__(self, llm_client: LLMClient, batch_size: int = 10, max_workers: int = 3,
                 circuit_breaker_threshold: int = 3):
        """
        初始化抽取器

        参数:
            llm_client: 已经实例化好的大语言模型客户端
            batch_size: 每个批次合并的文本块数量（默认10块）
            max_workers: 最大并发线程数（默认3个）
            circuit_breaker_threshold: 连续失败多少次触发熔断（默认3次）
        """
        self.llm_client = llm_client
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.circuit_breaker_threshold = circuit_breaker_threshold

        # ---- 熔断器状态（线程安全） ----
        self._consecutive_failures = 0
        self._failure_lock = threading.Lock()
        self._circuit_broken = False

        # ---- 最近一次抽取的统计数据（供前端读取） ----
        self.last_stats = {
            "total_batches": 0,
            "success_batches": 0,
            "failed_batches": 0,
            "total_triplets": 0,
        }

    @property
    def is_circuit_broken(self) -> bool:
        """供外部（如app.py）检查熔断器是否已触发"""
        return self._circuit_broken

    def _reset_circuit_breaker(self):
        """重置熔断器到初始状态（每次新任务开始时调用）"""
        with self._failure_lock:
            self._consecutive_failures = 0
            self._circuit_broken = False

    def _record_success(self):
        """记录一次成功请求，重置连续失败计数器"""
        with self._failure_lock:
            self._consecutive_failures = 0

    def _record_failure(self, batch_index: int):
        """
        记录一次失败请求，检查是否需要触发熔断

        参数:
            batch_index: 失败的批次编号（用于日志输出）
        """
        with self._failure_lock:
            self._consecutive_failures += 1
            current_failures = self._consecutive_failures
            if current_failures >= self.circuit_breaker_threshold:
                self._circuit_broken = True
                print(f"\n  🚨 熔断器触发！连续 {current_failures} 个批次请求失败！")
                print(f"     大模型API可能已失效，请检查 .env 中的配置。")

    def _extract_from_text(self, text: str, batch_index: int, max_retries: int = 3) -> list[dict]:
        """
        内部方法：从一段合并后的长文本中抽取三元组
        包含交错延迟、指数退避重试和熔断检查

        参数:
            text: 合并后的长文本
            batch_index: 当前批次的编号（从0开始）
            max_retries: 最大重试次数
        """
        # ---- 熔断检查：如果熔断器已触发，直接跳过 ----
        if self._circuit_broken:
            return []

        # ---- 交错发射：随机延迟0.3~0.8秒，错开并发请求的时间窗口 ----
        stagger_delay = random.uniform(0.3, 0.8)
        time.sleep(stagger_delay)

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
            # 每次重试前再检查一次熔断器
            if self._circuit_broken:
                return []

            response_text = self.llm_client.chat_with_json_output(user_message, system_prompt)

            # ---- 关键判断：None 代表 API 级别的故障 ----
            if response_text is None:
                if attempt < max_retries - 1:
                    # 指数退避：第1次等~1秒，第2次等~3秒，第3次等~6秒
                    backoff = (2 ** attempt) + random.uniform(0, 1)
                    print(f"  ⚠️ 批次 {batch_index + 1} API请求失败，等待 {backoff:.1f}秒后重试（第{attempt + 1}次）...")
                    time.sleep(backoff)
                    continue
                else:
                    # 所有重试都耗尽了
                    print(f"  ❌ 批次 {batch_index + 1} 经过 {max_retries} 次尝试仍失败。")
                    self._record_failure(batch_index)
                    return []

            # ---- 到这里说明 API 返回了有效的 JSON 字符串 ----
            try:
                data = json.loads(response_text)

                # 防御性编程 1：确保解析出来的是字典
                if not isinstance(data, dict):
                    print(f"  ⚠️ 批次 {batch_index + 1}: 大模型返回的根节点不是字典，跳过。")
                    self._record_failure(batch_index)
                    return []

                triplets = data.get("triplets", [])

                # 防御性编程 2：确保 triplets 是一个列表
                if not isinstance(triplets, list):
                    print(f"  ⚠️ 批次 {batch_index + 1}: triplets 字段不是列表格式，跳过。")
                    self._record_failure(batch_index)
                    return []

                # 防御性编程 3：过滤掉列表中不是字典的脏数据
                valid_triplets = []
                for t in triplets:
                    if isinstance(t, dict):
                        valid_triplets.append(t)
                    else:
                        print(f"  ⚠️ 丢弃格式错误的三元组数据 -> {t}")

                # ---- 成功！重置连续失败计数器 ----
                self._record_success()
                print(f"  ✅ 批次 {batch_index + 1} 成功抽取 {len(valid_triplets)} 个三元组。")
                return valid_triplets

            except json.JSONDecodeError:
                if attempt < max_retries - 1:
                    backoff = (2 ** attempt) + random.uniform(0, 1)
                    print(f"  ⚠️ 批次 {batch_index + 1} JSON解析失败，等待 {backoff:.1f}秒后重试（第{attempt + 1}次）...")
                    time.sleep(backoff)
                    continue
                print(f"  ❌ 批次 {batch_index + 1} 多次尝试解析JSON均失败，跳过。")
                self._record_failure(batch_index)
                return []

        return []

    def extract_triplets_from_chunks(self, chunks: list[dict], progress_callback=None) -> list[dict]:
        """
        核心对外接口：接收所有的文本块，执行智能节流并发抽取

        参数:
            chunks: 所有文档切片的列表
            progress_callback: 进度回调函数，用于更新前端进度条

        返回:
            所有成功抽取的三元组列表（如果熔断器触发，返回已抢救到的部分数据）
        """
        if not chunks:
            return []

        # ---- 重置状态 ----
        self._reset_circuit_breaker()

        # ---- 化零为整：将切片按 batch_size 打包 ----
        batches = []
        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i:i + self.batch_size]
            combined_text = "\n\n".join([c["content"] for c in batch_chunks])
            batches.append(combined_text)

        print(f"📦 共 {len(chunks)} 个文本块，已合并为 {len(batches)} 个批次准备抽取。")
        print(f"⚙️ 并发策略: {self.max_workers}线程 + 智能节流 + 熔断保护（阈值: 连续{self.circuit_breaker_threshold}次失败）")

        all_triplets = []
        completed_batches = 0
        failed_batches = 0

        # ---- 多线程并发处理（带智能节流） ----
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有批次任务，同时记录每个 future 对应的批次编号
            future_to_index = {
                executor.submit(self._extract_from_text, text, idx): idx
                for idx, text in enumerate(batches)
            }

            # as_completed 会在某个线程完成时立即返回
            for future in concurrent.futures.as_completed(future_to_index):
                batch_idx = future_to_index[future]

                try:
                    triplets = future.result()
                    if triplets:
                        all_triplets.extend(triplets)
                    else:
                        failed_batches += 1
                except Exception as e:
                    print(f"  批次 {batch_idx + 1} 发生系统错误: {e}")
                    failed_batches += 1

                completed_batches += 1
                if progress_callback:
                    progress_callback(completed_batches, len(batches))

                # ---- 熔断检查：如果熔断器已触发，立即取消所有未完成的任务 ----
                if self._circuit_broken:
                    print(f"\n  🛑 正在取消剩余 {len(batches) - completed_batches} 个待处理批次...")
                    for f in future_to_index:
                        if not f.done():
                            f.cancel()
                    break

        # ---- 保存统计数据（供前端读取） ----
        success_batches = completed_batches - failed_batches
        self.last_stats = {
            "total_batches": len(batches),
            "success_batches": success_batches,
            "failed_batches": failed_batches,
            "total_triplets": len(all_triplets),
        }

        # ---- 打印最终报告 ----
        print(f"\n📊 抽取报告：")
        print(f"   总批次: {len(batches)} | 成功: {success_batches} | 失败: {failed_batches}")
        print(f"   有效三元组: {len(all_triplets)} 个")

        if self._circuit_broken:
            print(f"   🚨 熔断器已触发，部分批次未处理！已抢救回 {len(all_triplets)} 个三元组。")
        elif failed_batches > 0:
            print(f"   ⚠️ 有 {failed_batches} 个批次抽取失败，图谱可能不完整。")
        else:
            print(f"   ✅ 全部批次处理成功！")

        return all_triplets


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    import config

    print("🚀 开始测试智能节流并发抽取...")

    if not config.LLM_API_KEY:
        print("❌ 请先在 .env 文件中配置 LLM_API_KEY")
    else:
        client = LLMClient(
            api_base=config.LLM_API_BASE,
            api_key=config.LLM_API_KEY,
            model_name=config.LLM_MODEL,
        )

        extractor = KGExtractor(client, batch_size=2, max_workers=2, circuit_breaker_threshold=3)

        # 模拟 4 个切片（会被打包成 2 个批次）
        test_chunks = [
            {"content": "RAG是一种结合信息检索和文本生成的技术。"},
            {"content": "LangChain框架完美支持了RAG技术的落地。"},
            {"content": "知识图谱可以与RAG结合形成GraphRAG。"},
            {"content": "Neo4j是最流行的图数据库，用于存储知识图谱。"}
        ]

        print(f"🧠 大模型正在使用智能节流并发抽取，请稍候...\n")

        triplets = extractor.extract_triplets_from_chunks(test_chunks)

        print("\n--- 抽取结果 ---")
        print(json.dumps(triplets, ensure_ascii=False, indent=2))

        print(f"\n--- 统计数据 ---")
        print(json.dumps(extractor.last_stats, ensure_ascii=False, indent=2))

        print(f"\n--- 熔断器状态 ---")
        print(f"是否触发: {extractor.is_circuit_broken}")
