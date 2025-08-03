import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from seed_data import load_data_from_folder, vector_store
from llm_gemini import func_retriever, get_llm

def setup_page():
    st.set_page_config(
        page_title="ChatBot_RAG",
        page_icon="💬",
        layout="wide"
    )

def initialize_app():
    load_dotenv()
    setup_page()

def setup_sidebar():
    with st.sidebar:
        st.title("⚙️Cấu hình")
        
        st.header("📑Embeddings Model")
        embeddings_choice = st.radio(
            "Chọn Embeddings Model:",
            ["models/embedding-001"]
        )

        st.header("🐧Model LLM")
        model_choice = st.radio(
            "Chọn AI Model để trả lời:",
            ["gemini-2.5-flash", "gemini-2.5-pro"]
        )

        st.header("📁Tải dữ liệu")
        uploaded_files = st.file_uploader(
            "Chọn một hoặc nhiều file dữ liệu .txt, docx, pdf",
            type=["txt", "docx", "pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            os.makedirs("data", exist_ok=True)
            for uploaded_file in uploaded_files:
                file_path = os.path.join("data", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
                st.success(f"✅Đã lưu file vào {file_path}")

        return embeddings_choice, model_choice

def setup_chat_interface(model_choice):
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("💬Chat-NVP")
    with col2:
        if st.button("🔄 Làm mới hội thoại"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Xin chào! Tôi có thể giúp gì cho bạn hôm nay?"}
            ]
            st.rerun()

    st.caption(f"🚀 Trợ lý AI được hỗ trợ bởi {model_choice}")

    msgs = StreamlitChatMessageHistory(key="langchain_messages")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Xin chào! Tôi có thể giúp gì cho bạn hôm nay?"}
        ]
        msgs.add_ai_message("Xin chào! Tôi có thể giúp gì cho bạn hôm nay?")

    # 👉 Hiển thị toàn bộ lịch sử hội thoại
    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "human"
        st.chat_message(role).write(msg["content"])

    return msgs

def process_user_message(msgs, retriever, model_choice):
    if prompt := st.chat_input("Hãy nhập câu hỏi của bạn..."):
        # Hiển thị câu hỏi
        st.session_state.messages.append({"role": "human", "content": prompt})
        with st.chat_message("human"):
            st.markdown(prompt)
        msgs.add_user_message(prompt)

        # Tạo callback và lấy lịch sử
        with st.chat_message("assistant"):
            
            last_response = ""
            for msg in reversed(st.session_state.messages[:-1]):
                if msg["role"] == "assistant":
                    last_response = msg["content"]
                    break

            # Ghép context từ câu trả lời gần nhất
            full_prompt = f"{last_response}\nCâu hỏi: {prompt}" if last_response else prompt

            # Gọi model  
            llm, final_prompt = get_llm(full_prompt, retriever, model_choice)
            st_callback = StreamlitCallbackHandler(st.container())
            response = llm.invoke(final_prompt, config={"callbacks": [st_callback]})
            answer = response.content if hasattr(response, "content") else str(response)
            st.markdown(answer)


        st.session_state.messages.append({"role": "assistant", "content": answer})
        msgs.add_ai_message(answer)

def main():
    initialize_app()
    embeddings_choice, model_choice = setup_sidebar()

    # Giao diện chat
    msgs = setup_chat_interface(model_choice)

    # Vector hóa dữ liệu
    all_chunks = load_data_from_folder()

    if not all_chunks:
        st.warning("📂Chưa có dữ liệu để tạo embeddings. Vui lòng upload file .txt vào sidebar.")
        return  # Dừng lại, không chạy tiếp main
    else:
        vector = vector_store(all_chunks, embeddings_choice)
        retriever = func_retriever(vector)

    # Chat
    process_user_message(msgs, retriever, model_choice)

if __name__ == "__main__":
    main()