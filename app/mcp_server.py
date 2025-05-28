import asyncio
from fastmcp import FastMCP
from qweather_tools import qweather_tool


mcp = FastMCP(
    name="weather",
    instructions="Integrate the qWeather API to provide a weather query tool for LLMs.",
)


@mcp.tool(
    name="get_weather_warning",
    description="根据提供的城市名查询天气预警信息。",
)
async def get_weather_warning(city: str):
    return await qweather_tool.get_weather_warning(city)


@mcp.tool(
    name="get_daily_forecast",
    description="根据提供的城市名，查询天气信息",
)
async def get_daily_forecast(city: str):
    return await qweather_tool.get_daily_forecast(city)


if __name__ == "__main__":
    # fastmcp run mcp_server.py:mcp
    asyncio.run(mcp.run())
