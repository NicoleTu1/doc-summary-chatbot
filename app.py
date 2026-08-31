import tempfile
import streamlit as st
import os
from backend import get_qa_chain, process_document

st.set_page_config(page_title="文件摘要與對話機器人", layout="centered")
st.title("📄 智能文件摘要與對話機器人")

# 設定環境變數或由使用者輸入 OpenAI API Key
os_environ = os.environ
if "OPENAI_API_KEY" not in os_environ:
    # 也可以透過介面輸入
    api_key = st.sidebar.text_input("請輸入 OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

# 檔案上傳區塊
uploaded_file = st.file_uploader("請上傳您的文件 (PDF 或 TXT)", type=["pdf", "txt"])

if uploaded_file is not None:
    # 將上傳的檔案寫入暫存檔供 LangChain 讀取
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
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