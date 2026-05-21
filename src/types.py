# ============================================================
# src/types.py - 核心类型定义
# ============================================================
"""ER 图中的核心数据类型：Tool, ToolCall, VectorStoreItem 等。"""

from dataclasses import dataclass, field


@dataclass
class Tool:
    """LLM function calling 工具定义（OpenAI 兼容格式）。

    对应 ER 图中 MCPClient 持有的 Tool 列表。
    """
    name: str
    description: str
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}})
    server_name: str = ""    # 所属 MCP Server 名称


@dataclass
class ToolCall:
    """LLM 返回的工具调用请求。

    对应 ER 图中 Agent 持有的 ToolCall。

    Attributes:
        id: 调用唯一标识
        name: 工具名称
        arguments: 解析后的参数字典
        raw_arguments: 原始 JSON 参数字符串
    """
    id: str
    name: str
    arguments: dict = field(default_factory=dict)
    raw_arguments: str = "{}"


@dataclass
class ToolResult:
    """工具执行结果。

    Attributes:
        tool_call_id: 对应的调用 ID
        tool_name: 工具名称
        content: 结果文本内容
        is_error: 是否为错误结果
    """
    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False


@dataclass
class LLMResponse:
    """LLM 单次调用响应。

    Attributes:
        content: 文本回复内容（可能为空）
        tool_calls: 工具调用列表（可能为 None）
        finish_reason: 结束原因 (stop / tool_calls / length)
    """
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    finish_reason: str = "stop"


# ── RAG 相关类型（后续使用） ─────────────────────────────

@dataclass
class VectorStoreItem:
    """向量存储条目。

    对应 ER 图中 VectorStore 包含的 VectorStoreItem。
    """
    embedding: list[float]
    document: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrieverResult:
    """检索结果。

    Attributes:
        query: 原始查询
        documents: 检索到的文档列表
        scores: 相似度分数列表（可选）
    """
    query: str
    documents: list[str]
    scores: list[float] = field(default_factory=list)