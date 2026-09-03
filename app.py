import io
import os
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt
import streamlit as st

from google import genai
from langchain_google_genai._common import GoogleGenerativeAIError

from app_config import FOOTER_CSS, PAGE_ICON, PAGE_TITLE, SIDEBAR_CSS, TITLE
from backend import (
    attach_inline_citations,
    build_citation_index,
    build_message_history,
    convert_file_to_text,
    get_qa_chain,
    md_to_text,
    process_document,
    release_retriever,
)


DEFAULT_FILE_PATH = Path(__file__).parent / "resume.md"


def initialize_session_state():
    """建立並保留目前 session 的對話與文件狀態。"""
    defaults = {
        "messages": [],
        "api_checked": False,
        "document_path": None,
        "document_key": None,
        "document_name": None,
        "qa_chain": None,
        "retriever": None,
        "default_file_requested": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def remove_temp_file(file_path):
    if file_path:
        try:
            Path(file_path).unlink(missing_ok=True)
        except OSError:
            pass


def clear_document_state():
    """釋放目前文件的向量庫、刪除暫存檔並清除文件狀態。"""
    release_retriever(st.session_state.get("retriever"))
    remove_temp_file(st.session_state.get("document_path"))

    st.session_state.retriever = None
    st.session_state.qa_chain = None
    st.session_state.document_path = None
    st.session_state.document_key = None
    st.session_state.document_name = None


def clear_session():
    clear_document_state()
    st.session_state.messages = []
    st.session_state.default_file_requested = False
    st.session_state.pop("uploaded_file", None)


def load_default_file():
    st.session_state.default_file_requested = True


def save_uploaded_file(uploaded_file):
    suffix = f"_{Path(uploaded_file.name).name}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
        temporary_file.write(uploaded_file.getvalue())
        return temporary_file.name


initialize_session_state()
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")

st.markdown("### " + TITLE)
st.caption(f"embeddings model `{os.getenv('embeddings_model')}` ‧ llm model `{os.getenv('llm_model')}`")

st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)
st.markdown(FOOTER_CSS, unsafe_allow_html=True)

if not st.session_state.api_checked:
    try:
        client = genai.Client()
        # 實務上最保險的檢查法：僅抓取模型清單，不消耗 Token 成本，用來驗證金鑰是否有效
        for model in client.models.list():
            pass

        st.toast("API 連線成功！", icon="✅")
    except Exception as e:
        error_message = str(e).lower()
        if "api key" in error_message or "authentication" in error_message:
            st.toast("API 金鑰無效或已過期，請檢查環境變數或 .env 設定", icon="⚠️")
        elif "quota" in error_message or "limit" in error_message:
            st.toast("API 額度已耗盡，請檢查 Google Cloud Console 的使用狀況", icon="⚠️")
        elif "resource" in error_message or "exhausted" in error_message:
            st.toast("目前使用量較高，請稍候後再試。", icon="⚠️")
        else:
            st.toast(f"API 連線異常：{str(e)}", icon="⚠️")
            
    # 關鍵：標記為已檢查，避免網頁後續重新整理時重複彈出
    st.session_state.api_checked = True


# 檔案上傳區塊

with st.sidebar:
    st.caption("此網頁僅作示範用途，請不要上傳太大的檔案，以免 API 使用額度耗盡，導致無法使用摘要與對話功能。")
    uploaded_file = st.file_uploader(
        "請上傳想摘要的文件 (PDF、TXT 或 MD) ⚠️ 請勿上傳敏感資料",
        type=["pdf", "txt", "md"],
        max_upload_size=5,
        help="此網頁串接 Google AI 模型來解析檔案，所以請不要上傳敏感資料。",
        key="uploaded_file",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.button(f"載入預設檔案 `{DEFAULT_FILE_PATH.name}`", 
                  on_click=load_default_file, 
                  disabled=not DEFAULT_FILE_PATH.exists(), 
                  type="primary", 
                  use_container_width=True)
    with col2:
        st.button("清除對話紀錄及檔案", on_click=clear_session, use_container_width=True)

    if uploaded_file is None and st.session_state.default_file_requested and DEFAULT_FILE_PATH.exists():
        default_file = io.BytesIO(DEFAULT_FILE_PATH.read_bytes())
        default_file.name = DEFAULT_FILE_PATH.name
        uploaded_file = default_file

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        document_key = (uploaded_file.name, hash(file_bytes))

        if document_key != st.session_state.document_key:
            clear_document_state()
            tmp_file_path = save_uploaded_file(uploaded_file)
            st.session_state.document_path = tmp_file_path
            st.session_state.document_key = document_key
            st.session_state.document_name = uploaded_file.name

            with st.spinner(f"正在對 `{uploaded_file.name}` 進行向量化與檔案解析..."):
                try:
                    st.session_state.retriever = process_document(tmp_file_path)
                    st.session_state.qa_chain = get_qa_chain(st.session_state.retriever)
                    st.write("檔案處理完畢，請在右側輸入框開始提問！")
                except Exception as error:
                    clear_document_state()
                    st.error(f"處理檔案時發生錯誤：{error}")

        st.subheader(f"檔案即時預覽：`{uploaded_file.name}`")
        if st.session_state.document_path:
            with st.container(height=500):
                preview_text, preview_lang = convert_file_to_text(Path(st.session_state.document_path))
                st.code(preview_text, language=preview_lang, wrap_lines=True, line_numbers=True)





# 渲染歷史對話
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收使用者輸入
if prompt := st.chat_input("請輸入您對文件的問題..."):
    if st.session_state.qa_chain is None:
        st.warning("請先上傳檔案並等待處理完成！")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("機器人思考中..."):
                chat_history = build_message_history(st.session_state.messages[:-1])
                try:
                    result = st.session_state.qa_chain.invoke({
                        "input": prompt,
                        "chat_history": chat_history
                    })

                    answer = result["answer"]
                    source_docs = result["context"]
                except GoogleGenerativeAIError as error:
                    st.error(f"Google API 伺服器回應異常 (500 Internal Error)，請稍後再試一次。  \n詳細錯誤: {error}")
                    st.stop()

                annotated_answer = attach_inline_citations(answer, source_docs)
                st.markdown(annotated_answer)
                st.session_state.messages.append({"role": "assistant", "content": annotated_answer})


                with st.expander("📌 檢視引用來源與對應片段 (Citations)"):
                    _, deduped_sources = build_citation_index(source_docs)
                    for i, source_group in enumerate(deduped_sources, 1):
                        source_name = source_group["source"]
                        pages = sorted({str(doc.metadata.get("page", "N/A")) for doc in source_group["docs"]})
                        page_display = ", ".join(pages) if pages else "N/A"

                        st.markdown(f"**來源 [{i}]：** `{source_name}` (頁碼/定位: `{page_display}`)")

                        snippets = []
                        for doc in source_group["docs"][:3]:
                            text = md_to_text(doc.page_content).strip()
                            if text:
                                snippets.append(text)

                        if snippets:
                            st.caption("\n\n".join(snippets))