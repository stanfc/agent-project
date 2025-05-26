from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from config import *
import os


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

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(all_docs)
    print(f"📚 總共切出 {len(chunks)} 段落")

    embedding = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL,
        query_instruction=QUERY_INSTRUCTION
    )

    db = FAISS.from_documents(chunks, embedding)
    db.save_local("vectorstore/faiss_index")
    print("✅ 向量索引已儲存完成")

if __name__ == "__main__":
    ingest_pdfs()
