# ============================================================
# src/agent.py - Agent 核心（ER 图对齐版 + RAG）
# ============================================================
"""Augmented LLM Agent — 编排 LLM + MCP + RAG。

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

import httpx

from .config import AppConfig
from .llm import DeepSeekLLM
from .mcp_client import MCPClient
from .mcp_manager import MCPManager
from .tool_executor import ToolExecutor
from .rag import EmbeddingRetriever
from .types import ToolCall, LLMResponse
from .logger import (
    banner_chat, banner_response, banner_tools,
    banner_tool_use, banner_tool_result, banner_mcp_connect,
    banner_error, banner_rag,
    log_info, log_warn, log_debug,
)


class Agent:
    """Augmented LLM Agent — Chat + MCP + RAG。"""

    DEFAULT_SYSTEM_PROMPT = (
        "你可以调用工具来完成任务，包括读写文件、列目录、抓取网页等。"
        "当需要使用工具时，请严格按 function calling 格式返回。"
        "使用中文回答用户问题。"
    )

    _MODEL_IDENTITY: dict[str, str] = {
        "deepseek-chat": "你是 DeepSeek，由深度求索公司创造的 AI 智能助手。",
        "deepseek-reasoner": "你是 DeepSeek-R1，由深度求索公司创造的推理模型。",
        "gpt-4o": "你是 GPT-4o，由 OpenAI 开发的 AI 助手。",
        "gpt-4": "你是 GPT-4，由 OpenAI 开发的 AI 助手。",
        "claude": "你是 Claude，由 Anthropic 开发的 AI 助手。",
        "qwen": "你是 Qwen（通义千问），由阿里云开发的 AI 助手。",
    }

    def __init__(self, config: AppConfig, system_prompt: str | None = None, context: str = ""):
        self.model: str = config.llm.model
        self.system_prompt: str = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.context: str = context

        self._config = config
        self._mcp_clients: list[MCPClient] = []
        self._mcp_manager = MCPManager(config.mcp_servers)
        self._tool_executor = ToolExecutor(self._mcp_manager)

        # ── RAG ──
        self._retriever: EmbeddingRetriever | None = None
        if config.embedding.api_key:
            self._retriever = EmbeddingRetriever(
                api_key=config.embedding.api_key,
                base_url=config.embedding.base_url,
                model=config.embedding.model,
            )

        self.llm: DeepSeekLLM | None = None

    # ── 公开方法 ──────────────────────────────────────────

    async def init(self) -> None:
        """初始化 LLM + MCP + RAG 索引。"""
        self.llm = DeepSeekLLM(
            config=self._config.llm,
            system_prompt=self._build_system_prompt(),
        )
        log_info(f"LLM 初始化: model={self.model}")

        await self._mcp_manager.connect_all()
        self._mcp_clients = list(self._mcp_manager.clients)

        # RAG: 构建向量索引
        await self._load_and_build_index()

        # 长期记忆
        memory_path = self._resolve_memory_path()
        if memory_path.exists():
            try:
                state = json.loads(memory_path.read_text(encoding="utf-8"))
                self.llm.load_state(state)
                log_info(f"记忆已恢复: {memory_path} ({self.llm.message_count} 条)")
            except Exception as e:
                log_info(f"记忆恢复失败: {e}")
                self.llm.reset_conversation()
        else:
            self.llm.reset_conversation()

        log_info("Agent 初始化完成 ✓")

    async def close(self) -> None:
        """关闭连接 + 保存记忆。"""
        log_info("Agent 正在关闭...")
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
        """统一入口：RAG 检索 → 注入上下文 → LLM。

        Args:
            prompt: 用户输入消息
        """
        if self.llm is None:
            raise RuntimeError("Agent 未初始化，请先调用 init()")

        # ── RAG 检索 ──
        augmented_prompt = await self._augment_prompt(prompt)

        banner_chat()
        log_info(f"用户消息: {augmented_prompt[:200]}{'...' if len(augmented_prompt) > 200 else ''}")

        tools = self._mcp_manager.get_tools_for_llm() if self._mcp_manager.is_connected else None
        last_response: LLMResponse | None = None

        for round_num in range(1, self._config.system.max_tool_rounds + 1):
            raw_response = await self.llm.chat(
                user_message=augmented_prompt if round_num == 1 else "",
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

            if last_response.content:
                banner_response()
                log_info(f"LLM 回复: {last_response.content[:200]}...")

            if not last_response.tool_calls:
                return last_response

            log_info(f"第 {round_num} 轮工具调用: {len(last_response.tool_calls)} 个")
            results = await self._tool_executor.execute_tool_calls([
                {
                    "id": tc.id,
                    "function": {"name": tc.name, "arguments": tc.raw_arguments},
                }
                for tc in last_response.tool_calls
            ])
            for result in results:
                self.llm.add_tool_result(
                    tool_call_id=result["tool_call_id"],
                    tool_name=result["tool_name"],
                    result_content=result["result"],
                )
            augmented_prompt = ""

        log_warn(f"达到最大工具调用轮数 ({self._config.system.max_tool_rounds})")
        return last_response or LLMResponse(content="(达到最大轮数)", finish_reason="stop")

    # ── 便捷方法 ──────────────────────────────────────────

    def get_tools_summary(self) -> list[dict[str, str]]:
        tools = self._mcp_manager.tools
        return [
            {"name": t["name"], "description": t.get("description", ""), "server": t.get("server_name", "?")}
            for t in tools
        ]

    def reset_conversation(self) -> None:
        if self.llm:
            self.llm.reset_conversation()
            try:
                self._resolve_memory_path().unlink(missing_ok=True)
            except Exception:
                pass

    def set_context(self, context: str) -> None:
        self.context = context
        if self.llm:
            self.llm.system_prompt = self._build_system_prompt()
            self.llm.reset_conversation()

    # ── RAG 内部方法 ─────────────────────────────────────

    async def _load_and_build_index(self) -> None:
        """加载数据源 → embedding → 构建向量索引。"""
        data_url = self._config.embedding.data_url
        if not data_url or not self._retriever:
            return

        log_info(f"RAG: 加载数据源 {data_url}")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(data_url)
                resp.raise_for_status()
                users: list[dict] = resp.json()
        except Exception as e:
            log_warn(f"RAG 数据加载失败: {e}")
            return

        log_info(f"RAG: 获取 {len(users)} 条数据，开始 embedding...")

        for i, user in enumerate(users):
            # 将每个 user 的关键信息拼成文档文本
            doc = json.dumps(user, ensure_ascii=False)
            try:
                self._retriever.embed_document(doc)
            except Exception as e:
                log_warn(f"RAG embedding 失败 (user {user.get('id', i)}): {e}")
                continue

        log_info(f"RAG: 索引构建完成 — {self._retriever.vector_store.size} 条")

    async def _augment_prompt(self, prompt: str) -> str:
        """对用户 prompt 做 RAG 检索并拼接上下文。"""
        if not self._retriever or self._retriever.vector_store.size == 0:
            return prompt

        banner_rag()
        try:
            results = self._retriever.retrieve(prompt, top_k=self._config.embedding.top_k)
        except Exception as e:
            log_warn(f"RAG 检索失败: {e}")
            return prompt

        if not results:
            log_info("RAG: 未检索到相关内容")
            return prompt

        log_info(f"RAG: 检索到 {len(results)} 条 (top score={results[0]['score']})")

        # 拼接检索到的用户信息
        context_parts = ["## 🔍 检索到的相关信息\n"]
        for i, r in enumerate(results):
            try:
                user = json.loads(r["document"])
                context_parts.append(
                    f"### 用户 {i+1}: {user.get('name', '?')} "
                    f"(@{user.get('username', '?')}) — 相似度 {r['score']}\n"
                    f"- 邮箱: {user.get('email', '?')}\n"
                    f"- 公司: {user.get('company', {}).get('name', '?')}\n"
                    f"- 地址: {user.get('address', {}).get('city', '?')}\n"
                )
            except (json.JSONDecodeError, KeyError):
                context_parts.append(f"### 条目 {i+1} (相似度 {r['score']})\n{r['document'][:200]}\n")

        context = "\n".join(context_parts)
        augmented = f"{prompt}\n\n{context}\n请根据以上检索到的用户信息回答问题。"

        log_info(f"RAG 上下文已注入: {len(context)} 字符")
        return augmented

    # ── 通用内部方法 ─────────────────────────────────────

    def _resolve_memory_path(self) -> Path:
        return Path(self._config.system.memory_file).resolve()

    def _build_system_prompt(self) -> str:
        identity = self._MODEL_IDENTITY.get(
            self.model,
            f"你是 {self.model} AI 智能助手。",
        )
        parts = [identity, "", self.system_prompt]
        if self.context:
            parts.append(f"\n## 参考上下文\n{self.context}")
        return "\n".join(parts)