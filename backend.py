from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda

import re
from pathlib import Path
from operator import itemgetter
import os



def process_document(file_path: str):
    """處理檔案：載入、切塊並建立向量資料庫與檢索器"""
    file_extension = Path(file_path).suffix.lower()
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_extension == ".md":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("不支援你所上傳的檔案類型，請上傳 PDF、TXT 或 MD 檔案。")

    docs = loader.load()
    
    # 進行文件切塊 (chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=60)
    splits = text_splitter.split_documents(docs)

    # 建立 Embedding 與向量資料庫
    embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv('embeddings_model'))
    vectorstore = Chroma.from_documents(documents=splits,embedding=embeddings)

    return vectorstore.as_retriever()

def format_docs(docs):
    """將檢索到的文件內容物件串接成純文字"""
    return "\n\n".join(doc.page_content for doc in docs)


def normalize_text_for_match(text: str) -> set:
    """將文字轉成可比較的 token 集合，方便計算句子與來源片段的相似度。"""
    if not text:
        return set()
    tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text.lower())
    return set(tokens)


def build_citation_index(source_docs: list):
    """將多個切塊按來源文件去除重複項，建立穩定的引用編號。"""
    if not source_docs:
        return {}, []

    source_order = []
    source_to_idx = {}

    for doc in source_docs:
        source_name = doc.metadata.get("source") or doc.metadata.get("file_name") or "unknown"
        source_name = Path(source_name).name    # 原本 source_name 是路徑, 只取最後一個檔名作為來源名稱
        if source_name not in source_to_idx:
            source_to_idx[source_name] = len(source_order) + 1
            source_order.append({
                "source": source_name,
                "docs": [doc],
                "pages": {doc.metadata.get("page", "N/A")},
            })
        else:
            source_order[source_to_idx[source_name] - 1]["docs"].append(doc)
            source_order[source_to_idx[source_name] - 1]["pages"].add(doc.metadata.get("page", "N/A"))

    return source_to_idx, source_order


def attach_inline_citations(answer: str, source_docs: list) -> str:
    """將回答中的句子貼上來源標記，例如「... [來源 1]」"""
    if not answer or not source_docs:
        return answer or ""

    source_to_idx, deduped_sources = build_citation_index(source_docs)

    def find_best_source_idx(text: str) -> int:
        best_idx = 0
        best_score = -1
        text_tokens = normalize_text_for_match(text)

        for source_name, idx in source_to_idx.items():
            docs = [doc for doc in source_docs if (doc.metadata.get("source") or doc.metadata.get("file_name") or "unknown") == source_name]
            for doc in docs:
                doc_tokens = normalize_text_for_match(doc.page_content)
                if not text_tokens or not doc_tokens:
                    continue
                score = len(text_tokens & doc_tokens) / max(len(text_tokens | doc_tokens), 1)
                if score > best_score:
                    best_score = score
                    best_idx = idx

        return best_idx if best_score > 0 else 0

    def annotate_text(text: str) -> str:
        text = text.strip()
        if not text:
            return ""

        match = re.match(r"^(\s*(?:[-*]|\d+\.)\s+)(.*)$", text, flags=re.DOTALL)
        if match:
            prefix, content = match.groups()
            idx = find_best_source_idx(content)
            if idx == 0:
                return text
            return f"{prefix}{content} `[來源 {idx}]`"

        idx = find_best_source_idx(text)
        if idx == 0:
            return text
        return f"{text} `[來源 {idx}]`"

    parsed_blocks = []
    for block in re.split(r"(\n\s*\n+)", answer.strip()):
        if not block:
            continue
        if re.fullmatch(r"\s*", block):
            parsed_blocks.append(block)
            continue

        if "\n" in block:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            parsed_blocks.extend(lines)
        else:
            parsed_blocks.append(block)

    annotated_sentences = []
    for block in parsed_blocks:
        stripped = block.strip()
        if not stripped:
            continue

        if re.match(r"^\s*(?:[-*]|\d+\.)\s+", stripped):
            annotated_sentences.append(annotate_text(stripped))
            continue

        sentences = re.split(r"(?<=[。.!?])\s+", stripped)
        if len(sentences) == 1:
            sentences = [stripped]

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            annotated_sentences.append(annotate_text(sentence))

    return "\n\n".join(part for part in annotated_sentences if part)


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



def sanitize_and_fallback(input_data):
    """
    input_data 是一個字典，包含:
    - "rewritten_query": LLM 改寫後的 Query
    - "original_input": 使用者的原始輸入
    """
    rewritten = input_data.get("rewritten_query", "").strip()
    original = input_data.get("original_input", "")
    
    # 1. 檢查是否為空、過短、或是異常過長 (例如超過 400 字元)
    if not rewritten or len(rewritten) < 2 or len(rewritten) > 400:
        print(f"[Warning] 改寫 Query 異常或過長，執行 Fallback。原文: {rewritten}")
        return original  # 降級使用原始輸入
        
    # 2. 過濾掉換行符號與多餘空白
    cleaned_query = re.sub(r'\s+', ' ', rewritten)
    
    # 3. 檢查是否包含惡意或不合法的特殊符號 (可依需求調整 Regex)
    # 這裡僅保留常見的文字、標點與數字
    # 若無大礙則回傳清理後的 Query
    return cleaned_query



def get_qa_chain(retriever):
    """
    使用現代 LCEL 建立問答鏈，並統一回傳 dict 格式：
    {
        "answer": str,
        "context": list[Document]
    }
    """
    llm = ChatGoogleGenerativeAI(model=os.getenv('llm_model'))

    # 1. 歷史查詢改寫鏈（處理上下文代名詞）
    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", "給定歷史對話與最新問題，請將問題改寫為一個不需依賴上下文也能獨立理解的完整查詢。若不需改寫則原封不動回傳。"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    query_transform_chain = rephrase_prompt | llm | StrOutputParser()

    query_transform_chain = (
        {
            "rewritten_query": query_transform_chain,
            "original_input": itemgetter("input")
        }
        | RunnableLambda(sanitize_and_fallback)
    )

    # 2. 檢索步驟：先改寫查詢，再從向量資料庫撈出對應的 Document 物件列表
    retrieval_step = (
        RunnablePassthrough.assign(
            rewritten_query=lambda x: query_transform_chain.invoke({
                "chat_history": x.get("chat_history", []),
                "input": x.get("input", ""),
            })
        )
        | RunnablePassthrough.assign(
            # 取得文件後，透過 format_docs 轉為字串，並同時保留原始 Document 物件供前端 Citations 使用
            context=lambda x: retriever.invoke(x["rewritten_query"])
        )
    )
    
    # 定義提示詞模板
    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system", 
            "請根據以下提供的上下文來回答問題。\n"
            "如果無法從上下文中找到答案，請誠實回答不知道、來源文件並沒有提供相關資訊，切勿妄加臆測。\n\n"
            "上下文內容：\n{context}"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # 組合 LCEL RAG pipeline
    # 讓輸出固定成 dict，方便前端讀取
    rag_chain = (
        retrieval_step
        | RunnablePassthrough.assign(
            # 在這裡將傳入 qa_prompt 的 context 轉為字串格式
            answer=(
                {
                    "context": lambda x: format_docs(x["context"]),
                    "chat_history": itemgetter("chat_history"),
                    "input": itemgetter("input"),
                }
                | qa_prompt 
                | llm 
                | StrOutputParser()
            ),
        )
        | (lambda x: {
            "answer": x["answer"],
            "context": x["context"], # 這裡保留原本的 Document 物件列表供前端渲染 Citations
        })
    )

    return rag_chain