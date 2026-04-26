"""
LLM（大语言模型）调用模块（最终加固版）

功能：
    提供统一的接口来调用各种兼容OpenAI格式的大模型API
    内置JSON清洗器，能处理大模型返回的各种"不规矩"格式
"""

import json
import re
from openai import OpenAI


def clean_json_response(text: str) -> str:
    """
    清洗大模型返回的JSON文本

    大模型经常在JSON外面套上Markdown代码块标记，比如：
        ```json
        {"key": "value"}
        ```
    或者在JSON前后加上多余的解释文字。
    这个函数会把所有这些"外衣"脱掉，只保留纯净的JSON字符串。

    参数:
        text: 大模型返回的原始文本

    返回:
        清洗后的纯JSON字符串
    """
    if not text:
        return "{}"

    cleaned = text.strip()

    # 策略1：用正则表达式匹配 ```json ... ``` 或 ``` ... ``` 中间的内容
    code_block_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    match = code_block_pattern.search(cleaned)
    if match:
        cleaned = match.group(1).strip()
        return cleaned

    # 策略2：如果没有代码块标记，尝试找到第一个 { 和最后一个 } 之间的内容
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace:last_brace + 1]
        return cleaned

    # 策略3：尝试找数组格式 [ ... ]
    first_bracket = cleaned.find("[")
    last_bracket = cleaned.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        cleaned = cleaned[first_bracket:last_bracket + 1]
        return cleaned

    # 如果什么都没找到，原样返回
    return cleaned


class LLMClient:
    """
    大语言模型客户端
    """

    def __init__(self, api_base: str, api_key: str, model_name: str):
        """
        初始化LLM客户端

        参数:
            api_base: API的基础地址
            api_key: API密钥
            model_name: 模型名称
        """
        self.model_name = model_name

        # 浏览器伪装头部，绕过部分第三方API的防火墙拦截
        custom_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }

        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key,
            default_headers=custom_headers,
        )

    def chat(self, user_message: str, system_prompt: str = None) -> str:
        """
        发送消息给LLM并获取回复

        参数:
            user_message: 用户的消息/问题
            system_prompt: 系统提示词（可选）

        返回:
            LLM的回复文本
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
            )
            reply = response.choices[0].message.content
            return reply

        except Exception as e:
            error_msg = f"调用LLM API时出错: {str(e)}"
            print(error_msg)
            return error_msg

    def chat_with_json_output(self, user_message: str, system_prompt: str = None) -> str:
        """
        发送消息并要求LLM返回JSON格式的回复
        内置三重保障机制确保输出合法JSON

        参数:
            user_message: 用户的消息
            system_prompt: 系统提示词（可选）

        返回:
            清洗后的JSON字符串
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_message})

        raw_response = ""

        # 第一次尝试：使用 response_format 强制JSON输出
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            raw_response = response.choices[0].message.content
        except Exception:
            # 第二次尝试：不用 response_format，在提示词中强调JSON
            fallback_system = (system_prompt or "") + "\n\n【重要】请务必只输出纯粹的JSON格式，不要用```包裹，不要输出任何解释文字。"
            messages_fallback = []
            if fallback_system:
                messages_fallback.append({"role": "system", "content": fallback_system})
            messages_fallback.append({"role": "user", "content": user_message})

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages_fallback,
                    temperature=0.1,
                )
                raw_response = response.choices[0].message.content
            except Exception as e:
                print(f"调用LLM API时出错: {str(e)}")
                return "{}"

        # 无论哪种方式拿到的结果，都过一遍清洗器
        cleaned = clean_json_response(raw_response)

        # 最终验证：尝试解析一下，确保是合法JSON
        try:
            json.loads(cleaned)
            return cleaned
        except json.JSONDecodeError:
            print(f"警告: 清洗后仍然不是合法JSON。大模型原始返回:\n{raw_response}")
            return "{}"


# ============================================================
# 测试代码
# ============================================================
if __name__== "__main__":
    import config

    if not config.LLM_API_KEY:
        print("错误: 请先在 .env 文件中配置 LLM_API_KEY")
    else:
        client = LLMClient(
            api_base=config.LLM_API_BASE,
            api_key=config.LLM_API_KEY,
            model_name=config.LLM_MODEL,
        )

        # 测试基本对话
        print("--- 测试基本对话 ---")
        reply = client.chat("你好，请用一句话介绍什么是知识图谱")
        print("回复:", reply)

        # 测试JSON输出
        print("\n--- 测试JSON输出 ---")
        json_reply = client.chat_with_json_output(
            "请从这句话中提取实体：LangChain框架支持RAG技术的落地应用。",
            '请提取实体并输出JSON格式：{"entities": ["实体1", "实体2"]}'
        )
        print("JSON回复:", json_reply)