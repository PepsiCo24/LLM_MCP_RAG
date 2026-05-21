# ============================================================
# main.py - 入口文件
# ============================================================
"""Augmented LLM 终端入口 — Chat + MCP + RAG。

ER 图流程:
    main.py → Agent.init() → Agent.invoke(prompt) → Agent.close()
"""

import asyncio
import sys

# 强制 UTF-8 输出（Windows 兼容）
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.agent import Agent
from src.logger import banner_error, log_info, log_warn, log_debug


# ============================================================
# 终端交互界面
# ============================================================

async def run_terminal(agent: Agent) -> None:
    """终端交互循环。"""
    print("\n" + "=" * 60)
    print("  🚀 Augmented LLM (Chat + MCP)")
    print(f"  模型: {agent.model}")
    print("=" * 60)
    print("  命令:")
    print("    输入消息      → 发送给 LLM")
    print("    /tools        → 查看可用工具")
    print("    /clear        → 清空对话历史")
    print("    /context <文本> → 设置 RAG 上下文")
    print("    exit / quit   → 退出")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        # ── 特殊命令 ──
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input == "/tools":
            _show_tools(agent)
            continue
        if user_input == "/clear":
            agent.reset_conversation()
            log_info("对话历史已清空")
            continue
        if user_input.startswith("/context "):
            ctx = user_input[len("/context "):]
            agent.set_context(ctx)
            log_info(f"RAG 上下文已设置: {ctx[:100]}...")
            continue

        # ── 调用 Agent ──
        try:
            response = await agent.invoke(user_input)
        except Exception as e:
            banner_error(f"调用失败: {e}")
            continue

        # 打印文本回复
        if response.content:
            print(f"\n🤖 助手: {response.content}")

        # 打印工具调用摘要
        if response.tool_calls:
            for tc in response.tool_calls:
                log_debug(f"已调用工具: {tc.name}")


def _show_tools(agent: Agent) -> None:
    """显示可用工具列表。"""
    tools = agent.get_tools_summary()
    if not tools:
        print("  (无可用工具 — 请检查 MCP Server 配置)")
        return
    print(f"\n  📦 可用工具 ({len(tools)}):")
    for t in tools:
        print(f"  • [{t['server']}] {t['name']}")
        print(f"    {t['description'][:80]}")


# ============================================================
# 入口
# ============================================================

async def main() -> None:
    """主函数。"""
    # 加载配置
    try:
        config = load_config("config.yaml")
    except FileNotFoundError as e:
        banner_error(str(e))
        print("  请确保 config.yaml 存在且配置正确")
        return
    except Exception as e:
        banner_error(f"配置加载失败: {e}")
        return

    # 检查 API Key
    if config.llm.api_key == "YOUR_DEEPSEEK_API_KEY":
        print("\n  ⚠️  请先在 config.yaml 中设置你的 DeepSeek API Key")
        print("     获取地址: https://platform.deepseek.com/api_keys")
        print("     配置文件: config.yaml → llm.api_key\n")
        return

    # 创建 Agent
    agent = Agent(config)

    try:
        # 初始化（连接 MCP Server + 初始化 LLM）
        await agent.init()
    except Exception as e:
        banner_error(f"Agent 初始化失败: {e}")
        return

    try:
        # 终端交互
        await run_terminal(agent)
    except KeyboardInterrupt:
        print("\n\n  已中断")
    finally:
        # 关闭 Agent
        await agent.close()
        print("\n  再见！👋\n")


if __name__ == "__main__":
    asyncio.run(main())