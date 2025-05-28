from langchain.agents import initialize_agent, Tool
from langchain.utilities import WikipediaAPIWrapper, SerpAPIWrapper
from langchain.tools import DuckDuckGoSearchRun



def get_search_tools():
    # 初始化工具
    wiki = WikipediaAPIWrapper()
    duck = DuckDuckGoSearchRun()

    # 包成 Tool 格式
    tools = [
        Tool.from_function(
            func=wiki.run,
            name="WikipediaSearch",
            description="Use this tool to search Wikipedia for factual, encyclopedia-style information."
        ),
        Tool.from_function(
            func=duck.run,
            name="DuckDuckGoSearch",
            description="Use this tool to search the web quickly using DuckDuckGo."
        )
    ]

    return tools