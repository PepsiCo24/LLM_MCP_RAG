# ============================================================
# src/mcp_client.py - 单个 MCP Server 客户端
# ============================================================
"""MCP (Model Context Protocol) 客户端 — JSON-RPC over stdio。"""

import asyncio
import json
import os
import sys
from typing import Any

from .logger import banner_mcp_connect, log_info, log_debug, log_warn, banner_error


class MCPClient:
    """与单个 MCP Server 通信的客户端。

    通过子进程 stdin/stdout 发送 JSON-RPC 请求。
    支持 initialize、tools/list、tools/call 等 MCP 协议方法。
    """

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
    ):
        self.name = name
        self.command = command
        self.args = args
        self.env = env or {}
        self.process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None
        self._server_info: dict[str, Any] | None = None

    async def connect(self) -> None:
        """启动 MCP Server 子进程并完成初始化握手。"""
        banner_mcp_connect(self.name)
        log_info(f"启动命令: {self.command} {' '.join(self.args)}")

        merged_env = os.environ.copy()
        merged_env.update(self.env)

        try:
            self.process = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=merged_env,
            )
        except FileNotFoundError:
            banner_error(f"找不到命令: {self.command}")
            raise
        except Exception as e:
            banner_error(f"启动失败: {e}")
            raise

        # 启动后台读取任务
        self._reader_task = asyncio.create_task(self._read_loop())

        # 发送 initialize 请求
        init_result = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "augmented-llm",
                "version": "0.1.0",
            },
        })
        self._server_info = init_result
        log_info(f"已连接: {self._server_info.get('serverInfo', {}).get('name', self.name)}")

        # 发送 initialized 通知
        await self._send_notification("notifications/initialized", {})

    async def disconnect(self) -> None:
        """关闭 MCP Server 连接。"""
        if self.process is not None:
            try:
                self.process.stdin.close()
            except Exception:
                pass
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            except Exception:
                pass
            self.process = None
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

    async def list_tools(self) -> list[dict[str, Any]]:
        """获取 Server 提供的工具列表。"""
        result = await self._send_request("tools/list", {})
        return result.get("tools", [])

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """调用指定工具。"""
        result = await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        # MCP 返回格式: { content: [{type: "text", text: "..."}] }
        return result

    # ── 内部方法 ──────────────────────────────────────────

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _send_request(self, method: str, params: dict) -> Any:
        """发送 JSON-RPC 请求并等待响应。"""
        req_id = self._next_id()
        request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future

        await self._write(request)
        log_debug(f"→ [{self.name}] {method} (id={req_id})")

        try:
            return await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise TimeoutError(f"MCP 请求超时: {method}")

    async def _send_notification(self, method: str, params: dict) -> None:
        """发送 JSON-RPC 通知（无需响应）。"""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        await self._write(notification)
        log_debug(f"→ [{self.name}] {method} (notification)")

    async def _write(self, data: dict) -> None:
        """写入一行 JSON 到子进程 stdin。"""
        if not self.process or not self.process.stdin:
            return
        line = json.dumps(data, ensure_ascii=False) + "\n"
        self.process.stdin.write(line.encode("utf-8"))
        await self.process.stdin.drain()

    async def _read_loop(self) -> None:
        """后台循环读取子进程 stdout 的 JSON-RPC 消息。"""
        if not self.process or not self.process.stdout:
            return
        try:
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    break
                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue
                try:
                    message = json.loads(line_str)
                except json.JSONDecodeError:
                    log_warn(f"[{self.name}] 非 JSON 输出: {line_str[:100]}")
                    continue

                await self._handle_message(message)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log_warn(f"[{self.name}] 读取异常: {e}")

    async def _handle_message(self, message: dict) -> None:
        """处理收到的 JSON-RPC 消息。"""
        msg_id = message.get("id")
        method = message.get("method", "")

        if msg_id is not None and "result" in message:
            # 响应
            future = self._pending.pop(msg_id, None)
            if future and not future.done():
                future.set_result(message["result"])
        elif msg_id is not None and "error" in message:
            # 错误响应
            future = self._pending.pop(msg_id, None)
            if future and not future.done():
                future.set_exception(Exception(
                    f"MCP Error: {message['error'].get('message', 'unknown')}"
                ))
        elif "method" in message:
            # 通知/服务器推送
            log_debug(f"← [{self.name}] notification: {method}")
        else:
            log_debug(f"← [{self.name}] 未知消息: {json.dumps(message, ensure_ascii=False)[:200]}")
