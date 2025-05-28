import functools
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI


def agent_node(state, agent, name):
    """将智能体转换成Node"""
    name = name.replace(" ", "_").replace("-", "_")
    result = agent.invoke(state)
    if not isinstance(result, ToolMessage):
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)

    return {
        "messages": [result],
        "sender": name,
    }
