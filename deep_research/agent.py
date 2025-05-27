from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationSummaryMemory
from langchain.memory import ConversationBufferMemory
from config import *
from tools import get_search_tools

instructions = """You are a tool-driven AI. Please strictly follow the response format below:

Thought: xxx
Action: tool name
Action Input: "input content"

You must carefully check whether the Observation already contains the answer to the question.
If the answer is clearly present in the Observation, extract it directly without performing further searches.

For all questions, you must perform a WebSearch first unless explicitly instructed not to search.
For the first web search, only use the simplest possible keywords — ideally containing only the nouns the user wants to know about, and nothing more.
Also, if you think something doesn't exist yet, please search it first instead of searching for other thing or search for rumor.
"""

def load_agent():

    
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
        streaming=True,
        convert_system_message_to_human=True
    )

    tools = get_search_tools()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={"system_message": instructions},
        verbose=True,
        memory=memory
    )

    return agent

def ask_question(agent, query: str):
    return agent.run(query)

if __name__ == "__main__":
    agent = load_agent()
    print("🌍 Deep Research Online Agent 啟動，輸入你的問題吧！（輸入 exit 結束）")

    while True:
        query = input("你問：")
        if query.lower() in ["exit", "quit"]:
            break
        response = agent.run(query)
        print("🤖 回答：", response)
