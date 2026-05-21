# ============================================================
# src/mcp_manager.py - 多 MCP Server 管理器
# ============================================================
"""管理多个 MCP Server 的连接、工具发现和聚合。"""

import asyncio
from typing import Any

from .config import MCPServerConfig
from .mcp_client import MCPClient
from .logger import banner_tools, log_info, log_warn, banner_error


class MCPManager:
    """管理多个 MCP Server 的生命周期和工具聚合。"""

    def __init__(self, server_configs: list[MCPServerConfig]):
        self._configs = server_configs
        self._clients: dict[str, MCPClient] = {}
        # 聚合后的工具列表: [{name, description, input_schema, server_name}, ...]
        self._tools: list[dict[str, Any]] = []

    @property
    def tools(self) -> list[dict[str, Any]]:
        """获取所有 Server 的聚合工具列表。"""
        return self._tools

    @property
    def clients(self) -> list:
        """获取所有已连接的 MCPClient 实例列表。"""
        return list(self._clients.values())

    @property
    def is_connected(self) -> bool:
        return len(self._clients) > 0

    async def connect_all(self) -> None:
        """连接所有配置的 MCP Server。"""
        if not self._configs:
            log_warn("未配置任何 MCP Server，跳过连接")
            return

        for cfg in self._configs:
            client = MCPClient(
                name=cfg.name,
                command=cfg.command,
                args=cfg.args,
                env=cfg.env,
            )
            try:
                await client.connect()
                self._clients[cfg.name] = client
            except Exception as e:
                banner_error(f"连接 {cfg.name} 失败: {e}")
                # 继续连接其他 server

        await self._refresh_tools()

    async def disconnect_all(self) -> None:
        """断开所有 MCP Server 连接。"""
        for name, client in self._clients.items():
            log_info(f"断开: {name}")
            await client.disconnect()
        self._clients.clear()
        self._tools.clear()

    async def _refresh_tools(self) -> None:
        """从所有 Server 刷新工具列表。"""
        banner_tools()
        self._tools.clear()

        for name, client in self._clients.items():
            try:
                tools = await client.list_tools()
                for tool in tools:
                    # 给每个工具标记来源 server
                    tool["server_name"] = name
                    self._tools.append(tool)
                log_info(f"[{name}] 发现 {len(tools)} 个工具:")
                for t in tools:
                    log_info(f"  • {t.get('name')} — {t.get('description', '(无描述)')[:60]}")
            except Exception as e:
                log_warn(f"[{name}] 获取工具列表失败: {e}")

        log_info(f"总计: {len(self._tools)} 个可用工具")

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """执行指定工具，自动路由到对应的 MCP Server。

        返回 MCP content 数组或原始结果。
        """
        # 查找工具所属的 server
        tool_info = None
        for t in self._tools:
            if t["name"] == tool_name:
                tool_info = t
                break

        if tool_info is None:
            raise ValueError(f"未找到工具: {tool_name}")

        server_name = tool_info["server_name"]
        client = self._clients.get(server_name)
        if client is None:
            raise RuntimeError(f"MCP Server 未连接: {server_name}")

        return await client.call_tool(tool_name, arguments)

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        """将 MCP 工具转换为 LLM function calling 格式 (OpenAI 兼容)。"""
        functions = []
        for tool in self._tools:
            func_def = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {
                        "type": "object",
                        "properties": {},
                    }),
                },
            }
            functions.append(func_def)
        return functions
