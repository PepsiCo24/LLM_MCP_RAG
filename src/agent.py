# ============================================================
# src/agent.py - Agent 核心（ER 图对齐版）
# ============================================================
"""Augmented LLM Agent — 编排 LLM + MCP 工具调用循环。

ER 图对应:
    Agent
    ├── mcpClients: MCPClient[]
    ├── llm: ChatOpenAI
    ├── model: string
    ├── systemPrompt: string
    ├── context: string
    ├── init()
    ├── close()
    └── invoke(prompt: string)
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from .config import AppConfig, MCPServerConfig
from .llm import DeepSeekLLM
from .mcp_client import MCPClient
from .mcp_manager import MCPManager
from .tool_executor import ToolExecutor
from .types import ToolCall, LLMResponse
from .logger import (
    banner_chat, banner_response, banner_tools,
    banner_tool_use, banner_tool_result, banner_mcp_connect,
    banner_error, banner_rag,
    log_info, log_warn, log_debug,
)


class Agent:
    """Augmented LLM Agent。

    对应 ER 图核心类，负责:
    - 管理多个 MCPClient 连接
    - 与 LLM 交互（ChatOpenAI）
    - 编排 tool-use 循环
    - invoke(prompt) 作为统一入口
    """

    # 默认系统提示词
    DEFAULT_SYSTEM_PROMPT = (
        "你可以调用工具来完成任务，包括读写文件、列目录、抓取网页等。"
        "当需要使用工具时，请严格按 function calling 格式返回。"
        "使用中文回答用户问题。"
    )

    # 常见模型的自我介绍映射
    _MODEL_IDENTITY: dict[str, str] = {
        "deepseek-chat": "你是 DeepSeek，由深度求索公司创造的 AI 智能助手。",
        "deepseek-reasoner": "你是 DeepSeek-R1，由深度求索公司创造的推理模型。",
        "gpt-4o": "你是 GPT-4o，由 OpenAI 开发的 AI 助手。",
        "gpt-4": "你是 GPT-4，由 OpenAI 开发的 AI 助手。",
        "claude": "你是 Claude，由 Anthropic 开发的 AI 助手。",
        "qwen": "你是 Qwen（通义千问），由阿里云开发的 AI 助手。",
    }

    def __init__(
        self,
        config: AppConfig,
        system_prompt: str | None = None,
        context: str = "",
    ):
        """初始化 Agent。

        Args:
            config: 应用配置
            system_prompt: 自定义系统提示词（可选）
            context: 额外上下文，注入到 system prompt 中（RAG 使用）
        """
        # ── ER 字段 ──
        self.model: str = config.llm.model
        self.system_prompt: str = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.context: str = context

        # ── 内部组件 ──
        self._config = config
        self._mcp_clients: list[MCPClient] = []
        self._mcp_manager = MCPManager(config.mcp_servers)
        self._tool_executor = ToolExecutor(self._mcp_manager)

        # LLM 延迟初始化（需要 API key）
        self.llm: DeepSeekLLM | None = None

    # ── 公开方法 ──────────────────────────────────────────

    async def init(self) -> None:
        """初始化：连接所有 MCP Server，初始化 LLM。

        对应 ER 图中 Agent.init()。
        """
        # 初始化 LLM
        self.llm = DeepSeekLLM(
            config=self._config.llm,
            system_prompt=self._build_system_prompt(),
        )
        log_info(f"LLM 初始化: model={self.model}")

        # 连接所有 MCP Server
        await self._mcp_manager.connect_all()

        # 更新 mcpClients 引用
        self._mcp_clients = list(self._mcp_manager.clients)

        # ── 长期记忆：尝试恢复，失败则初始化 ──
        memory_path = self._resolve_memory_path()
        if memory_path.exists():
            try:
                state = json.loads(memory_path.read_text(encoding="utf-8"))
                self.llm.load_state(state)
                log_info(f"记忆已恢复: {memory_path} ({self.llm.message_count} 条)")
            except Exception as e:
                log_info(f"记忆恢复失败: {e}，将重新初始化")
                self.llm.reset_conversation()
        else:
            self.llm.reset_conversation()

        log_info("Agent 初始化完成 ✓")

    async def close(self) -> None:
        """关闭：断开所有 MCP Server 连接。

        对应 ER 图中 Agent.close()。
        """
        log_info("Agent 正在关闭...")
        
        # ── 保存长期记忆 ──
        if self.llm:
            memory_path = self._resolve_memory_path()
            try:
                state = self.llm.save_state()
                memory_path.write_text(
                    json.dumps(state, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                log_info(f"记忆已保存: {memory_path} ({self.llm.message_count} 条)")
            except Exception as e:
                log_info(f"记忆保存失败: {e}")

        await self._mcp_manager.disconnect_all()
        self._mcp_clients.clear()
        log_info("Agent 已关闭 ✓")

    async def invoke(self, prompt: str) -> LLMResponse:
        """统一入口：发送用户消息，自动处理工具调用循环。

        对应 ER 图中 Agent.invoke(prompt: string)。

        Args:
            prompt: 用户输入消息

        Returns:
            LLMResponse: 包含文本回复和工具调用信息
        """
        if self.llm is None:
            raise RuntimeError("Agent 未初始化，请先调用 init()")

        banner_chat()
        log_info(f"用户消息: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")

        # 获取当前可用工具
        tools = self._mcp_manager.get_tools_for_llm() if self._mcp_manager.is_connected else None

        last_response: LLMResponse | None = None

        for round_num in range(1, self._config.system.max_tool_rounds + 1):
            # ── CHAT ──
            raw_response = await self.llm.chat(
                user_message=prompt if round_num == 1 else "",
                tools=tools,
            )

            last_response = LLMResponse(
                content=raw_response.get("content", ""),
                tool_calls=[
                    ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        raw_arguments=tc["function"]["arguments"],
                    )
                    for tc in (raw_response.get("tool_calls") or [])
                ],
                finish_reason=raw_response.get("finish_reason", "stop"),
            )

            # ── RESPONSE ──
            if last_response.content:
                banner_response()
                log_info(f"LLM 回复: {last_response.content[:200]}...")

            # 无工具调用 → 结束
            if not last_response.tool_calls:
                return last_response

            # ── TOOL USE ──
            log_info(f"第 {round_num} 轮工具调用: {len(last_response.tool_calls)} 个")
            results = await self._tool_executor.execute_tool_calls([
                {
                    "id": tc.id,
                    "function": {
                        "name": tc.name,
                        "arguments": tc.raw_arguments,
                    },
                }
                for tc in last_response.tool_calls
            ])

            # ── 注入工具结果到 LLM ──
            for result in results:
                self.llm.add_tool_result(
                    tool_call_id=result["tool_call_id"],
                    tool_name=result["tool_name"],
                    result_content=result["result"],
                )

            prompt = ""  # 后续轮次不传用户消息

        log_warn(f"达到最大工具调用轮数 ({self._config.system.max_tool_rounds})")
        return last_response or LLMResponse(content="(达到最大轮数)", finish_reason="stop")

    # ── 便捷方法 ──────────────────────────────────────────

    def get_tools_summary(self) -> list[dict[str, str]]:
        """获取当前可用工具摘要（名称 + 描述）。"""
        tools = self._mcp_manager.tools
        return [
            {"name": t["name"], "description": t.get("description", ""), "server": t.get("server_name", "?")}
            for t in tools
        ]

    def reset_conversation(self) -> None:
        """重置对话历史（同时删除持久化记忆文件）。"""
        if self.llm:
            self.llm.reset_conversation()
            memory_path = self._resolve_memory_path()
            try:
                memory_path.unlink(missing_ok=True)
                log_info("记忆文件已删除")
            except Exception:
                pass

    def set_context(self, context: str) -> None:
        """设置/更新 RAG 上下文（并刷新对话历史使上下文生效）。"""
        self.context = context
        if self.llm:
            self.llm.system_prompt = self._build_system_prompt()
            self.llm.reset_conversation()  # 用新 system prompt 重建 messages

    # ── 内部方法 ──────────────────────────────────────────

    def _resolve_memory_path(self) -> Path:
        """解析记忆文件的绝对路径。"""
        return Path(self._config.system.memory_file).resolve()

    def _build_system_prompt(self) -> str:
        """构建完整的 system prompt（动态注入模型身份 + RAG 上下文）。"""
        # 查找模型自我介绍
        identity = self._MODEL_IDENTITY.get(
            self.model,
            f"你是 {self.model} AI 智能助手。"
        )
        parts = [identity, "", self.system_prompt]
        if self.context:
            parts.append(f"\n## 参考上下文\n{self.context}")
        return "\n".join(parts)