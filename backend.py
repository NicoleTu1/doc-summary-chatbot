import os
from operator import itemgetter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pathlib import Path

from app_config import embeddings_model, llm_model



def process_document(file_path: str):
    """處理檔案：載入、切塊並建立向量資料庫與檢索器"""
    file_extension = Path(file_path).suffix.lower()
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("目前僅支援 PDF 與 TXT 檔案")

    docs = loader.load()
    
    # 進行文件切塊 (chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 建立 Embedding 與向量資料庫
    embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model)
    vectorstore = Chroma.from_documents(documents=splits,embedding=embeddings)

    return vectorstore.as_retriever()

def format_docs(docs):
    """將檢索到的文件內容物件串接成純文字"""
    return "\n\n".join(doc.page_content for doc in docs)

def get_qa_chain(retriever):
    """使用現代 LCEL 建立問答鏈"""
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm = ChatGoogleGenerativeAI(model=llm_model)
    
    # 定義提示詞模板
    prompt = ChatPromptTemplate.from_messages([
        (
            "system", 
            "請根據以下提供的上下文來回答問題。\n"
            "如果無法從上下文中找到答案，請誠實回答不知道，切勿妄加臆測。\n\n"
            "上下文內容：\n{context}"
        ),
        ("human", "{input}"),
    ])

    # 組合 LCEL RAG 管道：
    # 1. 取得使用者的 "input"，透過 retriever 找出相關文件，再用 format_docs 轉為字串填入 "context"
    # 2. 同時保留原本的 "input" 傳給提示詞模板
    # 3. 依序通過 prompt、llm 與 StrOutputParser（將輸出轉為純字串）
    rag_chain = (
        {
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain