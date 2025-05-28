"""多种tools的定义方式, 供llms调用"""
import asyncio
from typing import Annotated
from langchain_mcp_adapters.client import MultiServerMCPClient

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_mcp_adapters.tools import load_mcp_tools
from qweather_tools import qweather_tool


# Tavily 搜索工具，用于搜索
tavily_tool = TavilySearchResults(max_results=2)

# Python REPL 工具，用于执行 Python 代码
repl = PythonREPL()

@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"

    return f"Successfully executed:\n```python\n{code}\n```\n"


client = MultiServerMCPClient(
        {
            "weather": {
                "command": "python",
                "args": ["weather_mcp_server.py"],
                "transport": "stdio",
            }
            # "others": {
            #     # make sure you start your weather server on port 8000
            #     "url": "http://localhost:8000/sse",
            #     "transport": "sse",
            # }
        }
    )

async def weather_mcp_tools():
    return await client.get_tool("weather")

@tool(
    "get_weather_warning",
    description="根据提供的城市名查询天气预警信息。",
)
async def get_weather_warning(city: str):
    return await qweather_tool.get_weather_warning(city)

@tool(
    "get_daily_forecast",
    description="根据提供的城市名，和需要查询的天数，查询天气信息",
)
async def get_daily_forecast(city: str, days: int = 3):
    return await qweather_tool.get_daily_forecast(city, days)

tools = [tavily_tool, python_repl, get_daily_forecast, get_weather_warning]


if __name__ == "__main__":
    import os
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    chat_4o_model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        streaming=True,
        base_url="https://opnai-api.top/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    agent = create_react_agent(chat_4o_model, tools)
    print(asyncio.run(agent.ainvoke({"messages": "郑州的天气怎么样？"})))
