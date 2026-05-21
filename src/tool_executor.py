# ============================================================
# src/tool_executor.py - 工具执行器
# ============================================================
"""执行 LLM 请求的工具调用，负责解析参数、调用 MCP、返回结果。"""

import json
from typing import Any

from .mcp_manager import MCPManager
from .logger import banner_tool_use, banner_tool_result, log_info, log_warn, banner_error


class ToolExecutor:
    """工具执行器 — 将 LLM 的 tool_calls 路由到 MCP Server 执行。"""

    def __init__(self, mcp_manager: MCPManager):
        self._mcp = mcp_manager

    async def execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """批量执行工具调用。

        Args:
            tool_calls: LLM 返回的 tool_calls 列表
                [{id, function: {name, arguments}}]

        Returns:
            [{tool_call_id, tool_name, result}]
        """
        results = []
        for tc in tool_calls:
            result = await self._execute_single(tc)
            results.append(result)
        return results

    async def _execute_single(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """执行单个工具调用。"""
        call_id = tool_call["id"]
        func_info = tool_call["function"]
        tool_name = func_info["name"]
        arguments_str = func_info.get("arguments", "{}")

        banner_tool_use(tool_name)

        # 解析参数
        try:
            arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
        except json.JSONDecodeError:
            log_warn(f"参数解析失败: {arguments_str}")
            arguments = {}

        log_info(f"参数: {json.dumps(arguments, ensure_ascii=False, indent=2)}")

        # 执行工具
        try:
            raw_result = await self._mcp.execute_tool(tool_name, arguments)
        except Exception as e:
            banner_error(f"工具执行失败: {e}")
            return {
                "tool_call_id": call_id,
                "tool_name": tool_name,
                "result": f"Error: {e}",
            }

        banner_tool_result(tool_name)

        # 提取文本内容
        result_text = self._extract_text(raw_result)
        log_info(f"结果: {result_text[:300]}{'...' if len(result_text) > 300 else ''}")

        return {
            "tool_call_id": call_id,
            "tool_name": tool_name,
            "result": result_text,
        }

    def _extract_text(self, result: Any) -> str:
        """从 MCP 返回结果中提取文本内容。

        MCP 返回格式: { content: [{type: "text", text: "..."}] }
        """
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            content = result.get("content", [])
            if isinstance(content, list):
                texts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        texts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        texts.append(item)
                return "\n".join(texts) if texts else json.dumps(result, ensure_ascii=False)
            return json.dumps(result, ensure_ascii=False)
        if isinstance(result, list):
            texts = []
            for item in result:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif isinstance(item, str):
                    texts.append(item)
            return "\n".join(texts) if texts else json.dumps(result, ensure_ascii=False)
        return str(result)
