from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from config import *
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from langchain.tools import Tool

def write_file_tool(input_str: str):
    try:
        # 假設格式是 "filename|content"
        filename, content = input_str.split('|', 1)
        with open(filename.strip(), 'w', encoding='utf-8') as f:
            f.write(content.strip())
        return f"File '{filename.strip()}' written successfully!"
    except Exception as e:
        return f"Error writing file: {str(e)}"




def load_agent():
    embedding = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL,
        query_instruction=QUERY_INSTRUCTION
    )

    db = FAISS.load_local(
        "vectorstore/faiss_index",
        embedding,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.8,  
            "k": 10 
        }
    )
    '''
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 15, "fetch_k": 20, "lambda_mult": 0.5}
    )'''

    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )


    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
        convert_system_message_to_human=True
    )
    

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

    qa_tool = Tool(
        name="QA_System",
        func=lambda q: qa.run(q),
        description="Answer questions based on retrieved documents."
    )

    write_file = Tool(
        name="WriteFile",
        func=write_file_tool,
        description="Write content to a file. Format: filename|content"
    )

    tools = [qa_tool, write_file]
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent


def ask_question(agent, query: str):
    return agent.run(query)
