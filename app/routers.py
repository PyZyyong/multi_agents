from typing import Literal


def agent_router(state) -> Literal["call_tool", "__end__", "continue"]:
    """根据消息内容决定工作流走向"""
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "call_tool"

    if "FINAL ANSWER" in last_message.content:
        return "__end__"

    return "continue"