import os
from operator import itemgetter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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

def build_message_history(messages):
    """將 Streamlit 對話紀錄轉成 LangChain 的歷史訊息格式"""
    history = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")

        if role == "user":
            history.append(HumanMessage(content=content))
        elif role == "assistant":
            history.append(AIMessage(content=content))

    return history


def get_qa_chain(retriever):
    """
    使用現代 LCEL 建立問答鏈
    LCEL (LangChain Expression Language) 是 LangChain 推出的一套宣告式（Declarative）語法，
    專門用來將 AI 開發中的各種元件（如提示詞、大型語言模型、檢索器、輸出解析器）串接成一條完整的處理管線（Pipeline）。
    """
    llm = ChatGoogleGenerativeAI(model=llm_model)
    
    # 定義提示詞模板
    # 建立一個結構化的對話提示詞模板（Chat Prompt Template），
    # 專門用於 RAG（檢索增強生成）架構中，以規範大型語言模型（LLM）的回答行為。
    prompt = ChatPromptTemplate.from_messages([
        (
            "system", 
            "請根據以下提供的上下文來回答問題。\n"
            "如果無法從上下文中找到答案，請誠實回答不知道，切勿妄加臆測。\n"
            "如果問題和前面對話有關，請參考過去的對話內容來理解上下文。\n\n"
            "上下文內容：\n{context}"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # 組合 LCEL RAG 管道：
    # 1. 取得使用者的 "input"，透過 retriever 找出相關文件，再用 format_docs 轉為字串填入 "context"
    # 2. 同時保留「歷史訊息」與目前輸入，讓模型能記住過去對話
    # 3. 依序通過 prompt、llm 與 StrOutputParser（將輸出轉為純字串）
    rag_chain = (
        # ↓ LCEL 採用了類似 Linux 管道（Pipe）的 | 運算子。前一個元件的輸出會自動變成後一個元件的輸入。
        {
            "context": itemgetter("input") | retriever | format_docs,
            "chat_history": itemgetter("chat_history"),
            "input": itemgetter("input")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain