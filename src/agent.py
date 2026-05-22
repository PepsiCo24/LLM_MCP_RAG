# ============================================================
# src/agent.py - Agent 核心（ER 图对齐版 + RAG）
# ============================================================
"""Augmented LLM Agent — 编排 LLM + MCP + RAG。"""

import asyncio
import json
from datetime import datetime
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

    def __init__(self, config: AppConfig):
        self.model: str = config.llm.model
        self.system_prompt: str = self.DEFAULT_SYSTEM_PROMPT
        self._config = config
        self._mcp_clients: list[MCPClient] = []
        self._mcp_manager = MCPManager(config.mcp_servers)
        self._tool_executor = ToolExecutor(self._mcp_manager)

        # RAG
        self._retriever: EmbeddingRetriever | None = None
        self._rag_data_url: str = ""
        if config.embedding.api_key:
            self._retriever = EmbeddingRetriever(
                api_key=config.embedding.api_key,
                base_url=config.embedding.base_url,
                model=config.embedding.model,
            )

        self.llm: DeepSeekLLM | None = None

    # ================================================================
    # 公开方法
    # ================================================================

    async def init(self) -> None:
        """初始化 LLM + MCP + RAG 索引。"""
        self.llm = DeepSeekLLM(
            config=self._config.llm,
            system_prompt=self._build_system_prompt(),
        )
        log_info(f"LLM 初始化: model={self.model}")

        await self._mcp_manager.connect_all()
        self._mcp_clients = list(self._mcp_manager.clients)


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
        """统一入口：RAG 检索 → 注入上下文 → LLM。"""
        if self.llm is None:
            raise RuntimeError("Agent 未初始化，请先调用 init()")

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
                {"id": tc.id, "function": {"name": tc.name, "arguments": tc.raw_arguments}}
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

    async def reload_rag(self, url: str) -> str:
        """动态加载数据源：下载 → 保存 DATA/ → embedding → 重建索引。

        Returns:
            状态描述字符串
        """
        if not self._retriever:
            return "RAG 未启用（缺少 embedding.api_key）"

        self._rag_data_url = url
        log_info(f"RAG: 下载数据源 {url}")

        # 下载
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                items: list[dict] = resp.json()
        except Exception as e:
            msg = f"RAG 数据下载失败: {e}"
            log_warn(msg)
            return msg

        if not isinstance(items, list):
            return f"数据格式错误：期望 JSON 数组，实际为 {type(items).__name__}"

        # 保存到 DATA/
        data_dir = Path("DATA")
        data_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        saved = 0

        for item in items:
            name = item.get("username") or item.get("name") or item.get("id", "unknown")
            safe_name = str(name).replace("/", "_").replace("\\", "_")[:60]
            filepath = data_dir / f"{safe_name}.md"

            md = f"# {item.get('name', name)}\n\n"
            md += f"> 来源: {url}\n> 抓取时间: {timestamp}\n\n"
            md += "```json\n" + json.dumps(item, ensure_ascii=False, indent=2) + "\n```\n"
            filepath.write_text(md, encoding="utf-8")
            saved += 1

        log_info(f"RAG: 已保存 {saved} 条到 DATA/")

        # 重建向量索引
        self._retriever.vector_store = type(self._retriever.vector_store)()
        for item in items:
            doc = json.dumps(item, ensure_ascii=False)
            self._retriever.embed_document(doc)

        log_info(f"RAG: 索引重建完成 — {self._retriever.vector_store.size} 条")
        return f"RAG 数据已加载: {saved} 条 → DATA/ , 向量索引 {self._retriever.vector_store.size} 条"

    # ================================================================
    # 便捷方法
    # ================================================================

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

    # ================================================================
    # RAG 内部
    # ================================================================


    async def _augment_prompt(self, prompt: str) -> str:
        """RAG 检索 -> 调 LLM 生成摘要 -> 保存 RAG_Result/ -> 拼接上下文。"""
        if not self._retriever or self._retriever.vector_store.size == 0:
            return prompt

        banner_rag()
        top_k = self._config.embedding.top_k
        try:
            results = self._retriever.retrieve(prompt, top_k=top_k)
        except Exception as e:
            log_warn(f"RAG 检索失败: {e}")
            return prompt

        if not results:
            log_info("RAG: 未检索到相关内容")
            return prompt

        log_info(f"RAG: 检索到 {len(results)} 条 (top score={results[0]['score']})")
        log_info("RAG: 正在调 LLM 为 top_k 条数据生成个性化摘要...")

        # 异步生成摘要（不阻塞主流程）
        asyncio.create_task(self._generate_and_save_summaries(prompt, results))

        # 拼接上下文
        context_parts = ["## 检索到的相关信息\n"]
        for i, r in enumerate(results):
            try:
                user = json.loads(r["document"])
                context_parts.append(
                    f"### 用户 {i+1}: {user.get('name')} "
                    f"(@{user.get('username')}) - 相似度 {r['score']}\n"
                    f"- 邮箱: {user.get('email')}\n"
                    f"- 公司: {user.get('company', {}).get('name')}\n"
                    f"- 地址: {user.get('address', {}).get('city')}\n"
                )
            except (json.JSONDecodeError, KeyError):
                context_parts.append(f"### 条目 {i+1} (相似度 {r['score']})\n{r['document'][:200]}\n")

        context = "\n".join(context_parts)
        augmented = f"{prompt}\n\n{context}\n请根据以上检索到的用户信息回答问题。"
        log_info(f"RAG 上下文已注入: {len(context)} 字符")
        return augmented

    async def _generate_and_save_summaries(self, query: str, results: list[dict]) -> None:
        """调 LLM 为每条检索结果生成个性化摘要，保存到 RAG_Result/。"""
        if not self.llm or not self._retriever:
            return

        out_dir = Path("RAG_Result")
        out_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")

        users_text = ""
        for i, r in enumerate(results):
            try:
                u = json.loads(r["document"])
                users_text += f"### {i+1}. {u.get('name')} (@{u.get('username')}) [score={r['score']}]\n"
                c = u.get("company", {})
                a = u.get("address", {})
                users_text += f"company: {c.get('name')} | {c.get('catchPhrase')}\n"
                users_text += f"address: {a.get('city')}, {a.get('street')}\n"
                users_text += f"email: {u.get('email')} | phone: {u.get('phone')}\n\n"
            except Exception:
                users_text += f"### {i+1}. [score={r['score']}]\n{r['document'][:300]}\n\n"

        summary_prompt = (
            f"请为以下 {len(results)} 个人的数据各写一句80字以内的简体中文简介,"
            f"突出职业/公司/城市。必须每人一句，不可省略。"
            f"格式: @用户名: 简介\n\n{users_text}"
        )

        saved_messages = self.llm._messages.copy()
        try:
            raw = await self.llm.chat(summary_prompt, tools=None)
            summary_text = raw.get("content", "")
        except Exception as e:
            log_warn(f"RAG summary generation failed: {e}")
            return
        finally:
            self.llm._messages = saved_messages

        summaries = {}
        for line in summary_text.strip().split("\n"):
            line = line.strip()
            if line.startswith("@") and ":" in line:
                username, bio = line.split(":", 1)
                summaries[username.strip()] = bio.strip()

        # Aggregate MD
        summary_md = f"# RAG Results\n\n"
        summary_md += f"> Query: {query}\n> Time: {timestamp}\n> Source: {self._rag_data_url}\n\n---\n\n"

        for i, r in enumerate(results):
            try:
                u = json.loads(r["document"])
                username = u.get("username", "?")
                bio = summaries.get(f"@{username}", "(not generated)")
            except Exception:
                username = f"item_{i}"
                u = {}
                bio = "(parse error)"
            summary_md += f"## {u.get('name', username)} (@{username}) - Score {r['score']}\n\n"
            summary_md += f"**Bio**: {bio}\n\n"
            summary_md += f"- Email: {u.get('email', '?')}\n"
            summary_md += f"- Company: {u.get('company', {}).get('name', '?')}\n"
            summary_md += f"- City: {u.get('address', {}).get('city', '?')}\n\n---\n\n"

        (out_dir / f"rag_{ts_file}.md").write_text(summary_md, encoding="utf-8")

        # Individual user MDs
        for i, r in enumerate(results):
            try:
                u = json.loads(r["document"])
                username = u.get("username", f"item_{i}")
                bio = summaries.get(f"@{username}", "(not generated)")
                smd = f"# {u.get('name', username)}\n\n"
                smd += f"> Query: {query}\n> Score: {r['score']}\n> Time: {timestamp}\n\n"
                smd += f"**Bio**: {bio}\n\n"
                smd += f"- Email: {u.get('email', '?')}\n"
                smd += f"- Company: {u.get('company', {}).get('name', '?')}\n"
                smd += f"- City: {u.get('address', {}).get('city', '?')}\n"
                (out_dir / f"{username}.md").write_text(smd, encoding="utf-8")
            except Exception:
                pass

        log_info(f"RAG summaries saved: RAG_Result/ ({len(results)} items)")

    def _resolve_memory_path(self) -> Path:
        return Path(self._config.system.memory_file).resolve()

    def _build_system_prompt(self) -> str:
        identity = self._MODEL_IDENTITY.get(self.model, f"你是 {self.model} AI 智能助手。")
        return f"{identity}\n\n{self.system_prompt}"