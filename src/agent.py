# ============================================================
# src/agent.py - Agent ���ģ�Chat + MCP + RAG��
# ============================================================
import asyncio, json
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
    DEFAULT_SYSTEM_PROMPT = (
        "����Ե��ù�����������񣬰�����д�ļ�����Ŀ¼��ץȡ��ҳ�ȡ�"
        "����Ҫʹ�ù���ʱ�����ϸ� function calling ��ʽ���ء�"
        "ʹ�����Ļش��û����⡣"
    )
    _MODEL_IDENTITY = {
        "deepseek-chat": "���� DeepSeek�������������˾����� AI �������֡�",
        "deepseek-reasoner": "���� DeepSeek-R1�������������˾���������ģ�͡�",
        "gpt-4o": "���� GPT-4o���� OpenAI ������ AI ���֡�",
        "gpt-4": "���� GPT-4���� OpenAI ������ AI ���֡�",
        "claude": "���� Claude���� Anthropic ������ AI ���֡�",
        "qwen": "���� Qwen��ͨ��ǧ�ʣ����ɰ����ƿ����� AI ���֡�",
    }

    def __init__(self, config: AppConfig):
        self.model = config.llm.model
        self.system_prompt = self.DEFAULT_SYSTEM_PROMPT
        self._config = config
        self._mcp_clients: list[MCPClient] = []
        self._mcp_manager = MCPManager(config.mcp_servers)
        self._tool_executor = ToolExecutor(self._mcp_manager)
        self._retriever: EmbeddingRetriever | None = None
        self._rag_data_url = ""
        if config.embedding.api_key:
            self._retriever = EmbeddingRetriever(
                api_key=config.embedding.api_key,
                base_url=config.embedding.base_url,
                model=config.embedding.model,
            )
        self.llm: DeepSeekLLM | None = None

    # ================================================================
    # Public
    # ================================================================

    async def init(self):
        self.llm = DeepSeekLLM(config=self._config.llm, system_prompt=self._build_system_prompt())
        log_info(f"LLM init: model={self.model}")
        await self._mcp_manager.connect_all()
        self._mcp_clients = list(self._mcp_manager.clients)
        if self._config.embedding.data_url:
            await self.reload_rag(self._config.embedding.data_url)
        memory_path = self._resolve_memory_path()
        if memory_path.exists():
            try:
                state = json.loads(memory_path.read_text("utf-8"))
                self.llm.load_state(state)
                log_info(f"Memory restored: {memory_path} ({self.llm.message_count} msgs)")
            except Exception as e:
                log_info(f"Memory restore failed: {e}")
                self.llm.reset_conversation()
        else:
            self.llm.reset_conversation()
        log_info("Agent init done")

    async def close(self):
        log_info("Agent closing...")
        if self.llm:
            memory_path = self._resolve_memory_path()
            try:
                state = self.llm.save_state()
                memory_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
                log_info(f"Memory saved: {memory_path} ({self.llm.message_count} msgs)")
            except Exception as e:
                log_info(f"Memory save failed: {e}")
        await self._mcp_manager.disconnect_all()
        self._mcp_clients.clear()
        log_info("Agent closed")

    async def invoke(self, prompt: str) -> LLMResponse:
        if self.llm is None:
            raise RuntimeError("Agent not initialized")

        augmented_prompt = await self._augment_prompt(prompt)

        banner_chat()
        log_info(f"User: {prompt}")

        tools = self._mcp_manager.get_tools_for_llm() if self._mcp_manager.is_connected else None
        last_response = None

        for round_num in range(1, self._config.system.max_tool_rounds + 1):
            raw_response = await self.llm.chat(
                user_message=augmented_prompt if round_num == 1 else "",
                tools=tools,
            )
            last_response = LLMResponse(
                content=raw_response.get("content", ""),
                tool_calls=[
                    ToolCall(id=tc["id"], name=tc["function"]["name"], raw_arguments=tc["function"]["arguments"])
                    for tc in (raw_response.get("tool_calls") or [])
                ],
                finish_reason=raw_response.get("finish_reason", "stop"),
            )
            if last_response.content:
                banner_response()
            if not last_response.tool_calls:
                return last_response
            log_info(f"Tool round {round_num}: {len(last_response.tool_calls)} calls")
            results = await self._tool_executor.execute_tool_calls([
                {"id": tc.id, "function": {"name": tc.name, "arguments": tc.raw_arguments}}
                for tc in last_response.tool_calls
            ])
            for result in results:
                self.llm.add_tool_result(result["tool_call_id"], result["tool_name"], result["result"])
            augmented_prompt = ""
        log_warn(f"Max tool rounds ({self._config.system.max_tool_rounds})")
        return last_response or LLMResponse(content="(max rounds)", finish_reason="stop")

    async def reload_rag(self, url: str) -> str:
        if not self._retriever:
            return "RAG disabled (no embedding.api_key)"
        self._rag_data_url = url
        log_info(f"RAG downloading: {url}")
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                items: list[dict] = resp.json()
        except Exception as e:
            msg = f"RAG download failed: {e}"
            log_warn(msg)
            return msg
        if not isinstance(items, list):
            return f"Expected JSON array, got {type(items).__name__}"
        data_dir = Path("DATA")
        data_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for item in items:
            name = item.get("username") or item.get("name") or item.get("id", "unknown")
            safe = str(name).replace("/", "_").replace("\\", "_")[:60]
            md = f"# {item.get('name', name)}\n\n> Source: {url}\n> Time: {ts}\n\n"
            md += "```json\n" + json.dumps(item, ensure_ascii=False, indent=2) + "\n```\n"
            (data_dir / f"{safe}.md").write_text(md, encoding="utf-8")
        log_info(f"RAG saved {len(items)} to DATA/")
        self._retriever.vector_store = type(self._retriever.vector_store)()
        for item in items:
            self._retriever.embed_document(json.dumps(item, ensure_ascii=False))
        log_info(f"RAG index rebuilt: {self._retriever.vector_store.size}")
        return f"RAG loaded: {len(items)} -> DATA/, index {self._retriever.vector_store.size}"

    # ================================================================
    # Convenience
    # ================================================================

    def get_tools_summary(self):
        return [{"name": t["name"], "description": t.get("description", ""), "server": t.get("server_name", "?")}
                for t in self._mcp_manager.tools]

    def reset_conversation(self):
        if self.llm:
            self.llm.reset_conversation()
            try:
                self._resolve_memory_path().unlink(missing_ok=True)
            except Exception:
                pass

    # ================================================================
    # RAG internals
    # ================================================================

    async def _augment_prompt(self, prompt: str) -> str:
        if not self._retriever or self._retriever.vector_store.size == 0:
            return prompt
        banner_rag()
        top_k = self._config.embedding.top_k
        try:
            results = self._retriever.retrieve(prompt, top_k=top_k)
        except Exception as e:
            log_warn(f"RAG retrieve failed: {e}")
            return prompt
        if not results:
            log_info("RAG: no results")
            return prompt

        # --- Print all top_k detail to terminal ---
        sep = "=" * 50
        print(f"\n  {sep}")
        print(f"  RAG: top_{top_k} results ({len(results)} found):")
        for i, r in enumerate(results):
            try:
                u = json.loads(r["document"])
                c = u.get("company", {})
                a = u.get("address", {})
                print(f"  --- #{i+1} score={r['score']:.4f} ---")
                print(f"  Name:    {u.get('name','?')}  (@{u.get('username','?')})")
                print(f"  Email:   {u.get('email','?')}")
                print(f"  Phone:   {u.get('phone','?')}")
                print(f"  Website: {u.get('website','?')}")
                print(f"  Company: {c.get('name','?')}")
                print(f"           {c.get('catchPhrase','?')}")
                print(f"  Address: {a.get('city','?')}, {a.get('street','?')}, {a.get('suite','?')}")
                print(f"           {a.get('zipcode','?')}")
            except Exception:
                print(f"  --- #{i+1} score={r['score']:.4f} ---")
                print(f"  {r['document'][:120]}")
        print(f"  {sep}\n")

# --- Async: call LLM to generate summaries, save to RAG_Result/ ---
        await self._generate_summaries(prompt, results)

        # --- Build augmented prompt ---
        ctx = ["## RAG context (top_k={})".format(len(results))]
        for i, r in enumerate(results):
            try:
                u = json.loads(r["document"])
                ctx.append(f"### {i+1}. {u.get('name')} (@{u.get('username')}) score={r['score']:.4f}")
                ctx.append(f"- email: {u.get('email')} | company: {u.get('company',{}).get('name')} | city: {u.get('address',{}).get('city')}")
            except Exception:
                ctx.append(f"### {i+1}. score={r['score']:.4f}\n{r['document'][:200]}")
        context = "\n".join(ctx)
        augmented = f"{prompt}\n\n{context}\n\nReply using the above RAG context."
        log_info(f"RAG context injected: {len(context)} chars")
        return augmented

    async def _generate_summaries(self, query: str, results: list[dict]):
        if not self.llm or not self._retriever:
            return
        out_dir = Path("RAG_Result")
        out_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build data block
        users_text = ""
        for i, r in enumerate(results):
            try:
                u = json.loads(r["document"])
                c = u.get("company", {})
                a = u.get("address", {})
                users_text += f"### {i+1}. {u.get('name')} (@{u.get('username')}) [score={r['score']:.4f}]\n"
                users_text += f"company: {c.get('name')} | {c.get('catchPhrase')}\n"
                users_text += f"address: {a.get('city')}, {a.get('street')}\n"
                users_text += f"email: {u.get('email')} | phone: {u.get('phone')}\n\n"
            except Exception:
                users_text += f"### {i+1}. [score={r['score']:.4f}]\n{r['document'][:300]}\n\n"

        summary_prompt = (
            f"����һλ���ﴫ�����ҡ���������� {len(results)} ���˵����ݣ�"
            f"Ϊÿ��׫дһ�� 150-200 �ֵĸ��Ի���飨�������ģ���Ҫ�й��¸С�"
            f"������������ְҵ��ݣ����company��catchPhrase�������ڳ��С�"
            f"�Լ���������Ϣ�������������ϸ�ڻ���������"
            f"�����ʽ��ÿ���� @�û��� ��ͷ�����зָ��\n\n{users_text}"
        )

        saved = self.llm._messages.copy()
        try:
            raw = await self.llm.chat(summary_prompt, tools=None)
            text = raw.get("content", "")
        except Exception as e:
            log_warn(f"RAG summary gen failed: {e}")
            return
        finally:
            self.llm._messages = saved

        # Parse summaries
        summaries = {}
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("@") and ":" in line:
                username, bio = line.split(":", 1)
                summaries[username.strip()] = bio.strip()

        # Aggregate MD
        agg = f"# RAG Results\n\n> Query: {query}\n> Time: {ts}\n> Source: {self._rag_data_url}\n\n---\n\n"
        for i, r in enumerate(results):
            try:
                u = json.loads(r["document"])
                username = u.get("username", "?")
                bio = summaries.get(f"@{username}", "(not generated)")
            except Exception:
                username = f"item_{i}"
                u = {}
                bio = "(parse error)"
            agg += f"## {u.get('name', username)} (@{username}) �� Score {r['score']:.4f}\n\n"
            agg += f"**Abstract**: {bio}\n\n"
            agg += "```json\n" + json.dumps(u, ensure_ascii=False, indent=2)[:1500] + "\n```\n\n---\n\n"
        (out_dir / f"rag_{ts_file}.md").write_text(agg, encoding="utf-8")

        # Individual MDs
        for i, r in enumerate(results):
            try:
                u = json.loads(r["document"])
                username = u.get("username", f"item_{i}")
                bio = summaries.get(f"@{username}", "(not generated)")
                smd = f"# {u.get('name', username)}\n\n"
                smd += f"> Query: {query}\n> Score: {r['score']:.4f}\n> Time: {ts}\n> Source: {self._rag_data_url}\n\n"
                smd += f"## Abstract\n\n{bio}\n\n"
                smd += "## Raw Data\n\n```json\n" + json.dumps(u, ensure_ascii=False, indent=2) + "\n```\n"
                (out_dir / f"{username}.md").write_text(smd, encoding="utf-8")
            except Exception:
                pass

        log_info(f"RAG summaries saved: RAG_Result/ ({len(results)} items)")

    # ================================================================
    # Internal
    # ================================================================

    def _resolve_memory_path(self) -> Path:
        return Path(self._config.system.memory_file).resolve()

    def _build_system_prompt(self) -> str:
        identity = self._MODEL_IDENTITY.get(self.model, f"���� {self.model} AI �������֡�")
        return f"{identity}\n\n{self.system_prompt}"
