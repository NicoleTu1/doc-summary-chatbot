import tempfile
import streamlit as st

from google import genai

import os
import time

from app_config import PAGE_TITLE, PAGE_ICON, TITLE, FOOTER_CSS, embeddings_model, llm_model
from backend import get_qa_chain, process_document

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")

# st.title(TITLE)
# st.header(TITLE)
st.markdown("### " + TITLE)
st.caption(f"embeddings model: {embeddings_model}  \nllm model: {llm_model}")

st.markdown(FOOTER_CSS, unsafe_allow_html=True)

if "api_checked" not in st.session_state:
    st.session_state.api_checked = False
if not st.session_state.api_checked:
    with st.status("正在初始化 API...", expanded=True) as status:
        try:
            client = genai.Client()
            # 實務上最保險的檢查法：僅抓取模型清單，不消耗 Token 成本，用來驗證金鑰是否有效
            for model in client.models.list(): pass
            
            st.toast("API 連線成功！後台狀態正常 🟢", icon="✅")
        except Exception as e:
            # 補捉金鑰無效、額度耗盡或網路不通的錯誤
            st.toast(f"API 連線異常：{str(e)}", icon="⚠️")
            
    # 關鍵：標記為已檢查，避免網頁後續重新整理時重複彈出
    st.session_state.api_checked = True




# 設定環境變數或由使用者輸入 API Key
# api_type = "GOOGLE_API_KEY" # OPENAI_API_KEY
# api_key = os.getenv(api_type)
# if not api_key:
#     api_key = st.secrets[api_type]
# 也可以透過介面輸入
# api_key = st.sidebar.text_input("請輸入 API Key", type="password")


# 檔案上傳區塊
uploaded_file = st.file_uploader("請上傳想摘要的文件 (PDF 或 TXT)", 
                                 type=["pdf", "txt"], 
                                 help="⚠️ 請勿上傳敏感資料 ⚠️ \n此網頁僅作示範用途，若 API 使用額度耗盡，會造成無法使用摘要與對話功能。")

if uploaded_file is not None:
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
                # 直接呼叫 LCEL 鏈，回傳結果即為解答字串
                answer = st.session_state.qa_chain.invoke({"input": prompt})
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})