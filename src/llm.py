# ============================================================
# src/llm.py - DeepSeek LLM 封装（ER 图 ChatOpenAI 对应实现）
# ============================================================
"""使用 OpenAI 兼容 SDK 调用 DeepSeek 模型。

ER 图对应:
    ChatOpenAI
    ├── llm: OpenAI
    ├── model: string
    ├── messages: OpenAI.Chat.ChatCompletionMessageParam[]
    ├── tools: Tool[]
    ├── chat(prompt?: string)
    └── appendToolResult(toolCallId: string, toolOutput: string)
"""

from typing import Any

from openai import AsyncOpenAI

from .config import LLMConfig
from .logger import banner_chat, banner_response, log_info, log_debug


class DeepSeekLLM:
    """DeepSeek LLM 封装 — 支持 Chat + Function Calling。

    对应 ER 图中 ChatOpenAI，使用 DeepSeek API (OpenAI 兼容)。
    """

    def __init__(self, config: LLMConfig, system_prompt: str | None = None):
        self.config = config
        self.model: str = config.model
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.system_prompt: str = system_prompt or self._build_default_prompt()
        self._messages: list[dict[str, Any]] = []

    @staticmethod
    def _build_default_prompt() -> str:
        """根据配置自动生成包含模型身份的系统提示词。"""
        return (
            "你可以调用工具来完成任务，包括读写文件、列目录、抓取网页等。"
            "当需要使用工具时，请严格按 function calling 格式返回。"
            "使用中文回答用户问题。"
        )

    def reset_conversation(self) -> None:
        """重置对话历史。"""
        self._messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        log_info("对话历史已清空")
    def save_state(self) -> dict[str, Any]:
        """导出当前状态（用于持久化）。"""
        return {
            "system_prompt": self.system_prompt,
            "messages": self._messages,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """从持久化状态恢复。"""
        self.system_prompt = state.get("system_prompt", self.system_prompt)
        self._messages = state.get("messages", [])
        log_info(f"记忆已恢复: {len(self._messages)} 条消息")

    @property
    def message_count(self) -> int:
        """当前消息数量。"""
        return len(self._messages)

    async def chat(
        self,
        user_message: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """发送消息给 DeepSeek，返回响应。

        对应 ER 图中 ChatOpenAI.chat(prompt?: string)。

        Args:
            user_message: 用户输入（空字符串表示仅让 LLM 继续处理工具结果）
            tools: function calling 工具定义列表

        Returns:
            {
                "role": "assistant",
                "content": str | None,
                "tool_calls": list | None,
                "finish_reason": str,
            }
        """
        banner_chat()
        if user_message:
            log_info(f"用户消息: {user_message[:200]}{'...' if len(user_message) > 200 else ''}")

        # 添加用户消息到历史
        if user_message:
            self._messages.append({"role": "user", "content": user_message})

        # 构建请求参数
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # 调用 API
        response = await self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        banner_response()

        # 提取文本内容
        content = message.content or ""

        # 提取工具调用
        tool_calls = None
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

        result = {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
            "finish_reason": choice.finish_reason,
        }

        # 将 assistant 消息加入历史
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if content:
            assistant_msg["content"] = content
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
                for tc in tool_calls
            ]
        self._messages.append(assistant_msg)

        # 打印响应摘要
        if content:
            log_info(f"LLM 文本回复: {content[:300]}{'...' if len(content) > 300 else ''}")
        if tool_calls:
            for tc in tool_calls:
                log_info(f"请求工具: {tc['function']['name']}({tc['function']['arguments'][:100]}...)")

        return result

    def add_tool_result(
        self, tool_call_id: str, tool_name: str, result_content: str
    ) -> None:
        """将工具执行结果添加到对话历史。

        对应 ER 图中 ChatOpenAI.appendToolResult(toolCallId, toolOutput)。
        """
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result_content,
        })
        log_debug_suffix = f" ({result_content[:80]}...)" if len(result_content) > 80 else ""
        log_debug(f"工具结果已注入: {tool_name}{log_debug_suffix}")