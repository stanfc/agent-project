from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import Document
from config import *
import os
import re

def latex_block_splitter(text, chunk_size=500, chunk_overlap=50):
    # 先提取所有 \begin{}...\end{} block
    begin_end_pattern = re.compile(r'(\\begin\{.*?\}.*?\\end\{.*?\})', re.DOTALL)
    blocks = begin_end_pattern.split(text)

    chunks = []
    for block in blocks:
        if begin_end_pattern.match(block):
            # 是 \begin{}...\end{} 區塊，直接保留為一段
            chunks.append(block.strip())
        else:
            # 處理剩下的部分
            command_pattern = re.compile(r'(\\[^\s]+)')
            parts = command_pattern.split(block)
            temp = ""
            for part in parts:
                if command_pattern.match(part):
                    # 是 \ 開頭指令，直接當一段
                    if temp:
                        chunks.append(temp.strip())
                        temp = ""
                    chunks.append(part.strip())
                else:
                    # 普通文字，分段組裝
                    temp += part
                    if len(temp) >= chunk_size:
                        chunks.append(temp.strip())
                        temp = ""
            if temp:
                chunks.append(temp.strip())

    # 最後加上 overlap
    final_chunks = []
    for i in range(0, len(chunks), 1):
        combined = ""
        for j in range(max(0, i - chunk_overlap), min(len(chunks), i + 1)):
            combined += chunks[j] + "\n"
        final_chunks.append(combined.strip())

    print(f"✅ 共切出 {len(final_chunks)} 段落")
    return final_chunks


def ingest_pdfs():
    all_docs = []

    pdf_dir = "data/pdfs"
    for fname in os.listdir(pdf_dir):
        if fname.endswith(".pdf"):
            path = os.path.join(pdf_dir, fname)
            print(f"📄 載入 PDF：{fname}")
            loader = PyPDFLoader(path)
            docs = loader.load()
            all_docs.extend(docs)

    #splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    # 把所有 document 的內容串起來
    combined_text = "\n".join(doc.page_content for doc in all_docs)

    chunks = latex_block_splitter(combined_text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    print(f"📚 總共切出 {len(chunks)} 段落")

    embedding = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL,
        query_instruction=QUERY_INSTRUCTION
    )
    

    chunk_docs = [Document(page_content=chunk) for chunk in chunks]

    db = FAISS.from_documents(chunk_docs, embedding)
    db.save_local("vectorstore/faiss_index")
    print("✅ 向量索引已儲存完成")

if __name__ == "__main__":
    ingest_pdfs()
