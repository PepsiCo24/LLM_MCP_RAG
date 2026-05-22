# ============================================================
# src/config.py - 配置管理
# ============================================================
"""加载 YAML 配置并提供类型化访问。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LLMConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    max_tokens: int = 4096
    temperature: float = 0.7


@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class EmbeddingConfig:
    api_key: str = ""
    base_url: str = "https://api.siliconflow.cn/v1"
    model: str = "BAAI/bge-m3"
    data_url: str = ""
    top_k: int = 3


@dataclass
class SystemConfig:
    max_tool_rounds: int = 10
    log_level: str = "INFO"
    memory_file: str = "chat_memory.json"


@dataclass
class AppConfig:
    llm: LLMConfig
    mcp_servers: list[MCPServerConfig]
    embedding: EmbeddingConfig
    system: SystemConfig


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """从 YAML 文件加载配置。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    llm = LLMConfig(**raw.get("llm", {}))

    servers = []
    for s in raw.get("mcp_servers", []):
        servers.append(MCPServerConfig(
            name=s["name"],
            command=s["command"],
            args=s.get("args", []),
            env=s.get("env", {}),
        ))

    embedding = EmbeddingConfig(**raw.get("embedding", {}))

    system = SystemConfig(**raw.get("system", {}))

    return AppConfig(llm=llm, mcp_servers=servers, embedding=embedding, system=system)