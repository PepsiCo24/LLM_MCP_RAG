# ============================================================
# main.py - 入口文件
# ============================================================
"""Augmented LLM 终端入口 — Chat + MCP + RAG。"""

import asyncio
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.agent import Agent
from src.logger import banner_error, log_info, log_warn, log_debug


async def run_terminal(agent: Agent) -> None:
    """终端交互循环。"""
    print("\n" + "=" * 60)
    print("  🚀 Augmented LLM (Chat + MCP + RAG)")
    print(f"  模型: {agent.model}")
    print("=" * 60)
    print("  命令:")
    print("    输入消息        → 发送给 LLM")
    print("    /tools          → 查看可用工具")
    print("    /clear          → 清空对话历史 + 记忆 + RAG 数据")
    print("    /RAG <url>      → 加载数据源 URL（重建向量索引）")
    print("    exit / quit     → 退出")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            break
        if user_input == "/tools":
            _show_tools(agent)
            continue
        if user_input == "/clear":
            agent.reset_conversation()
            log_info("已清空：对话历史 + 记忆 + DATA/ + RAG_Result/")
            continue
        if user_input.startswith("/RAG "):
            url = user_input[len("/RAG "):].strip()
            if url:
                log_info(f"正在加载 RAG 数据: {url}")
                result = await agent.reload_rag(url)
                print(f"\n  📡 {result}")
            continue

        try:
            response = await agent.invoke(user_input)
        except Exception as e:
            banner_error(f"调用失败: {e}")
            continue

        if response.content:
            print(f"\n🤖 助手: {response.content}")

        if response.tool_calls:
            for tc in response.tool_calls:
                log_debug(f"已调用工具: {tc.name}")


def _show_tools(agent: Agent) -> None:
    tools = agent.get_tools_summary()
    if not tools:
        print("  (无可用工具)")
        return
    print(f"\n  📦 可用工具 ({len(tools)}):")
    for t in tools:
        print(f"  • [{t['server']}] {t['name']}")
        print(f"    {t['description'][:80]}")


async def main() -> None:
    try:
        config = load_config("config.yaml")
    except FileNotFoundError as e:
        banner_error(str(e))
        return
    except Exception as e:
        banner_error(f"配置加载失败: {e}")
        return

    if config.llm.api_key == "YOUR_DEEPSEEK_API_KEY":
        print("\n  ⚠️  请先在 config.yaml 中设置你的 DeepSeek API Key")
        print("     获取地址: https://platform.deepseek.com/api_keys\n")
        return

    agent = Agent(config)

    try:
        await agent.init()
    except Exception as e:
        banner_error(f"Agent 初始化失败: {e}")
        return

    try:
        await run_terminal(agent)
    except KeyboardInterrupt:
        print("\n\n  已中断")
    finally:
        await agent.close()
        print("\n  再见！👋\n")


if __name__ == "__main__":
    asyncio.run(main())