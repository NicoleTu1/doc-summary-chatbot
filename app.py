import tempfile
import streamlit as st
import os
from backend import get_qa_chain, process_document

st.set_page_config(page_title="文件摘要與對話機器人 | Nicole", layout="centered")

# st.title("📄 文件摘要與對話機器人")
# st.header("📄 文件摘要與對話機器人")
st.markdown("### 📄 文件摘要與對話機器人")

footer_css = """
<style>
.custom-footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    
    background-color: var(--background-color); 
    color: var(--text-color);
    
    text-align: center;
    padding: 8px 8px 8px 8px; /* 調整頁尾的內邊距: 上右下左 */
    font-size: 14px;
    z-index: 999;   /* 讓頁尾置於最上層 */
    
    border-top: 1px solid rgba(190, 190, 190, 0.5); 
}
.custom-footer a {
    /* link color */
    color: var(--primary-color); 
    text-decoration: none;
}
</style>
<div class="custom-footer">
    Made by <b>Nicole</b> | nicoletuatie@gmail.com
    <!-- <a href="https://linkedin.com" target="_blank">LinkedIn</a> -->
</div>
"""
# 3. 渲染頁尾（必須設定 allow_html=True）
st.markdown(footer_css, unsafe_allow_html=True)


# 設定環境變數或由使用者輸入 API Key
api_type = "GOOGLE_API_KEY" # OPENAI_API_KEY
api_key = os.getenv(api_type)
if not api_key:
    api_key = st.secrets[api_type]
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