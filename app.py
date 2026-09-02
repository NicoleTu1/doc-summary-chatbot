import io
import os
import tempfile
from pathlib import Path
from markdown_it import MarkdownIt
import streamlit as st

from google import genai
from langchain_google_genai._common import GoogleGenerativeAIError


from app_config import PAGE_TITLE, PAGE_ICON, TITLE, FOOTER_CSS
from backend import *

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")

st.markdown("### " + TITLE)
st.caption(f"embeddings model `{os.getenv('embeddings_model')}` | llm model `{os.getenv('llm_model')}`")

st.markdown(FOOTER_CSS, unsafe_allow_html=True)

if "api_checked" not in st.session_state:
    st.session_state.api_checked = False
if not st.session_state.api_checked:
    try:
        client = genai.Client()
        # 實務上最保險的檢查法：僅抓取模型清單，不消耗 Token 成本，用來驗證金鑰是否有效
        for model in client.models.list():
            pass

        st.toast("API 連線成功！後台狀態正常 🟢", icon="✅")
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


md = MarkdownIt()
def md_to_text(text: str) -> str:
    tokens = md.parse(text)

    result = []

    for token in tokens:
        if token.type == "inline":
            result.append(token.content)

    return "\n".join(result)




# 檔案上傳區塊
st.caption("此網頁僅作示範用途，請不要上傳太大的檔案，以免 API 使用額度耗盡，導致無法使用摘要與對話功能。", text_alignment="center")

DEFAULT_UPLOADED_FILE = Path(r"./tests/test_input/resume.md")

if DEFAULT_UPLOADED_FILE.exists():
    default_bytes = DEFAULT_UPLOADED_FILE.read_bytes()
    default_file = io.BytesIO(default_bytes)
    default_file.name = DEFAULT_UPLOADED_FILE.name
else:
    default_file = None

uploaded_file = st.file_uploader(
    "請上傳想摘要的文件 (PDF、TXT 或 MD) ⚠️ 請勿上傳敏感資料 ",
    type=["pdf", "txt", "md"],
    max_upload_size = 5,  # 限制上傳檔案大小為 5 MB
    help="此網頁串接 Google AI 模型來解析檔案，所以請不要上傳敏感資料。",
    key="uploaded_file",
)

if uploaded_file is None and default_file is not None:
    uploaded_file = default_file

if uploaded_file is not None:
    if default_file is not None and uploaded_file is default_file:
        st.info("目前已載入預設範例檔案：resume.md")

    # if st.button("預覽目前檔案內容", use_container_width=True):
        raw_bytes = uploaded_file.getvalue()
        try:
            preview_text = raw_bytes.decode("utf-8")
            preview_lang = "markdown"
        except UnicodeDecodeError:
            preview_text = raw_bytes.decode("utf-8", errors="replace")
            preview_lang = "text"

    with st.popover("檔案預覽", use_container_width=True):
        st.code(preview_text[:20000], language=preview_lang, wrap_lines=True, line_numbers=True)
    # 將上傳的檔案寫入暫存檔供 LangChain 讀取
    # 為了後面能使用 process_document(), 而該函式需要檔案路徑作為參數, 因此這裡使用 tempfile 產生暫存檔, 才能產生路徑.
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
        # ↑ delete=False 代表離開 with 區塊後，不要自動刪除這個暫存檔。
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    with st.spinner("正在進行向量化與檔案解析..."):
        try:
            retriever = process_document(tmp_file_path)
            st.session_state.qa_chain = get_qa_chain(retriever)
            st.success("檔案處理完畢，請在下方輸入框開始提問！")
        except Exception as e:
            st.error(f"處理檔案時發生錯誤：{e}")



# 初始化對話紀錄
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染歷史對話
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收使用者輸入
if prompt := st.chat_input("請輸入您對文件的問題..."):
    if "qa_chain" not in st.session_state:
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
                except GoogleGenerativeAIError as e:
                    st.error(f"Google API 伺服器回應異常 (500 Internal Error)，請稍後再試一次。  \n詳細錯誤: {e}")

                # st.markdown(result["answer"])
                # st.session_state.messages.append({"role": "assistant", "content": answer})
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