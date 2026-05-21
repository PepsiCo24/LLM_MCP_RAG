# ============================================================
# src/logger.py - 日志横幅系统
# ============================================================
"""统一的日志横幅打印，用于标识各个处理阶段。"""

import sys
from datetime import datetime

# 强制 UTF-8 输出（Windows 兼容）
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 终端颜色（ANSI）
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _banner(label: str, color: str = CYAN, width: int = 60) -> None:
    """打印带颜色的横幅日志。"""
    now = datetime.now().strftime("%H:%M:%S")
    banner_line = f"{'=' * width}"
    label_line = f"  {label}"
    timestamp_line = f"  [{now}]"

    print(f"\n{color}{BOLD}{banner_line}{RESET}")
    print(f"{color}{BOLD}{label_line}{RESET}")
    print(f"{color}  {timestamp_line}{RESET}")
    print(f"{color}{BOLD}{banner_line}{RESET}\n")
    sys.stdout.flush()


def banner_chat() -> None:
    """=== CHAT === — 用户消息送入 LLM 时"""
    _banner("=== CHAT ===", CYAN)


def banner_response() -> None:
    """=== RESPONSE === — LLM 返回文本回复时"""
    _banner("=== RESPONSE ===", GREEN)


def banner_tools() -> None:
    """=== TOOLS === — 发现/列出可用工具时"""
    _banner("=== TOOLS ===", BLUE)


def banner_tool_use(tool_name: str) -> None:
    """=== TOOL USE === — 准备调用某个工具时"""
    _banner(f"=== TOOL USE: {tool_name} ===", YELLOW)


def banner_tool_result(tool_name: str) -> None:
    """=== TOOL RESULT === — 工具返回结果时"""
    _banner(f"=== TOOL RESULT: {tool_name} ===", MAGENTA)


def banner_mcp_connect(server_name: str) -> None:
    """=== MCP CONNECT === — 连接 MCP Server 时"""
    _banner(f"=== MCP CONNECT: {server_name} ===", BLUE)


def banner_rag() -> None:
    """=== RAG === — RAG 检索时"""
    _banner("=== RAG ===", MAGENTA)


def banner_error(msg: str) -> None:
    """=== ERROR === — 出错时"""
    _banner(f"=== ERROR: {msg} ===", RED)


def log_info(msg: str) -> None:
    """普通信息日志。"""
    print(f"  [INFO] {msg}")


def log_debug(msg: str) -> None:
    """调试日志。"""
    print(f"  [DEBUG] {msg}")


def log_warn(msg: str) -> None:
    """警告日志。"""
    print(f"{YELLOW}  [WARN] {msg}{RESET}")