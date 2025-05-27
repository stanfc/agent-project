from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from config import *
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

embedder = SentenceTransformer(EMBEDDING_MODEL)

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

    '''retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.8,  # 你要設定的相似度門檻，越高越嚴格（通常 0.5～0.8）
            "k": 10               # 至多取幾個（符合門檻的前 k 個）
        }
    )'''
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 15, "fetch_k": 20, "lambda_mult": 0.5}
    )

    
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
    return qa


def ask_question(agent, query: str):
    return agent.run(query)
