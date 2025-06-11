import os
import asyncio
import operator
from typing import TypedDict, Annotated, Sequence
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph_supervisor import create_supervisor
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from mcp_adapters import tavily_tool, python_repl, get_weather_warning, get_daily_forecast

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "default"

chat_4o_model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    streaming=True,
    base_url="https://opnai-api.top/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)

deepseek_model = ChatDeepSeek(
    # deepseek-chat 对应 DeepSeek-V3；deepseek-reasoner 对应 DeepSeek-R1。
    model="deepseek-chat",
    temperature=0.2,
    streaming=True,
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


def create_agent(llm, tools, tool_message: str, custom_notice: str = ""):
    """创建一个智能体。"""
    # 定义智能体的提示模板，包含系统消息和工具信息
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个AI助手，可以与其他助手合作，一起帮助用户解决问题。"
                "如果你无法独立完成回答，可以将任务交给其他助手继续处理。"
                "如果你有满足用户需求的最终结果，请在响应前加上 FINAL ANSWER 以便程序停止。"
                "\n{custom_notice}\n"
                "你有以下工具可以使用: {tool_names}.\n{tool_message}\n\n",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # 将系统消息部分和工具名称插入到提示模板中
    prompt = prompt.partial(tool_message=tool_message, custom_notice=custom_notice)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))

    return prompt | llm.bind_tools(tools)

checkopint = InMemorySaver()

research_agent = create_react_agent(
    model=deepseek_model,
    tools=[tavily_tool],
    prompt=f"""你是一个研究员助手，对于不了解的问题可以使用以下工具进行搜索：{tavily_tool.name}，\n
    在使用搜索工具之前，请仔细思考并明确查询内容。然后，进行一次搜索，一次性解决查询的所有需求。""",
    name="research_assistant"
)

chart_agent = create_react_agent(
    model=deepseek_model,
    tools=[python_repl],
    prompt=f"""你是一个专业图表生成专家，基于其他智能助手提供的数据，生成图表。要求图表清晰，易于理解。\n
    你有以下工具使用:{python_repl.name}""",
    name="chart_assistant"
)

weather_agent = create_react_agent(
    model=deepseek_model,
    tools=[get_weather_warning, get_daily_forecast],
    prompt="""你是一个智能天气查询助手，核心任务是通过调用内置工具获取实时数据，并转化为结构清晰的图表结构，\n
    图表结构内应该详细的记录天气信息的各个指标，帮助用户理解。需保持专业且口语化的表达，必要时用符号/表情辅助理解（如🌤️⛈️）。
    **你可以使用的工具**
    get_weather_warning: 根据提供的城市名查询天气预警信息。
    get_daily_forecast: 根据提供的城市名，查询最近日期的天气信息如一周、三天内、五天内、一个月等。
    """,
    name="weather_assistant"
)

supervisor_agent = create_supervisor(
    agents=[chart_agent, weather_agent, research_agent],
    model=deepseek_model,
    prompt=(
        """你是一个AI助手管理员，管理以下助手：weather_assistant，chart_assistant， research_assistant。仔细分析用户的需求，如果你无法独立完成回答，可以将任务交给其他助手继续处理。\n
        比如用户要查询天气信息，你应该使用weather_assistant，用户要生成图表你应该使用chart_assistant，对于你不确定的问题，你应该使用research_assistant，\n
        你可以同时使用一个或者多个助手协作完成任务。"""
    ),
    supervisor_name='manager',
    output_mode="full_history"
)

supervisor = supervisor_agent.compile(checkpointer=checkopint)

from IPython.display import display, Image
display(
    Image(
        supervisor.get_graph(xray=True).draw_mermaid_png(max_retries=3)
    )
)
if __name__ == "__main__":
    # print(chat_4o_model.invoke("你好"))
    async def main():
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            async for chunk in supervisor.astream(
                {"messages": ("user", user_input)},
                config={"configurable": {"thread_id": "current_user_id"}}
            ):
                sender = {'manager', 'chart_assistant', 'weather_assistant', 'research_assistant'}
                for s in sender:
                    if messages := chunk.get(s):
                        break
                if messages:
                    last_message = messages['messages'][-1]
                    if not isinstance(last_message, ToolMessage):
                        # print(last_message.pretty_print())
                        print(last_message)

    asyncio.run(main())
