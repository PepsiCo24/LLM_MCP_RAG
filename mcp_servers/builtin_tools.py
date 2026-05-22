# ============================================================
# mcp_servers/builtin_tools.py - 内置测试 MCP Server
# ============================================================
"""内置 MCP Server，提供文件操作和网页抓取工具。

运行方式: .venv/Scripts/python mcp_servers/builtin_tools.py
依赖: pip install mcp httpx
"""

import re
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

# ── 创建 FastMCP Server ──────────────────────────────────

server = FastMCP("augmented-llm-builtin-tools")


# ── 工具: 读取文件 ───────────────────────────────────────

@server.tool()
async def read_file(path: str) -> str:
    """读取本地文件内容。

    Args:
        path: 文件路径（绝对路径或相对路径）
    """
    filepath = Path(path)
    if not filepath.exists():
        return f"错误: 文件不存在 — {filepath}"
    try:
        content = filepath.read_text(encoding="utf-8")
        return content
    except Exception as e:
        return f"错误: 读取失败 — {e}"


# ── 工具: 写入文件 ───────────────────────────────────────

@server.tool()
async def write_file(path: str, content: str) -> str:
    """将内容写入本地文件。

    Args:
        path: 文件路径
        content: 要写入的内容
    """
    filepath = Path(path)
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
        return f"成功写入: {filepath} ({len(content)} 字符)"
    except Exception as e:
        return f"错误: 写入失败 — {e}"


# ── 工具: 列出目录 ───────────────────────────────────────

@server.tool()
async def list_directory(path: str = ".") -> str:
    """列出目录中的文件。

    Args:
        path: 目录路径
    """
    dirpath = Path(path)
    if not dirpath.exists():
        return f"错误: 目录不存在 — {dirpath}"
    try:
        items = []
        for item in sorted(dirpath.iterdir()):
            item_type = "DIR" if item.is_dir() else "FILE"
            items.append(f"  [{item_type}] {item.name}")
        return "\n".join(items) if items else "(空目录)"
    except Exception as e:
        return f"错误: 列出失败 — {e}"


# ── 工具: 网页抓取并总结 ─────────────────────────────────

def _html_to_text(html: str) -> str:
    """简单 HTML → 纯文本转换。"""
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@server.tool()
async def web_fetch_summary(url: str, max_length: int = 2000) -> str:
    """抓取网页内容，生成摘要并自动保存为 Markdown 文件到 abstract/ 目录。

    Args:
        url: 网页 URL
        max_length: 摘要最大字符数
    """
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            response.raise_for_status()
            html = response.text
    except Exception as e:
        return f"错误: 网页抓取失败 — {e}"

    # 提取标题
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    page_title = title_match.group(1).strip() if title_match else None

    # HTML → 纯文本
    text = _html_to_text(html)
    summary = text[:max_length]

    # ── 保存到 abstract/ ──
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 文件名：优先用标题，否则用域名
    if page_title:
        safe_title = re.sub(r"[\\/:*?\"<>|]", "_", page_title)[:60]
        filename = f"{safe_title}.md"
    else:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.replace(":", "_").replace(".", "_")
        filename = f"{domain}.md"

    out_dir = Path("abstract")
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename

    md_content = f"""# {page_title or url}

> 来源: [{url}]({url})
> 抓取时间: {timestamp}
> 原始长度: {len(text)} 字符

---

{summary}
"""
    filepath.write_text(md_content, encoding="utf-8")
    save_msg = f"已保存到: {filepath}"

    return f"来源: {url}\n标题: {page_title or "(无)"}\n原始长度: {len(text)} 字符\n{save_msg}\n\n内容摘要:\n\n{summary}"


# ── 启动 ─────────────────────────────────────────────────

if __name__ == "__main__":
    server.run()