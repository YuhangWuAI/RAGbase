import streamlit as st
import os
import logging
from langchain_community.llms import Ollama
from document_loader import load_documents_into_database
from model_utils import get_list_of_models
from llm_chain import get_streaming_chain
from config import DEFAULT_EMBEDDING_MODEL, DEFAULT_PATH, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.title("FinQA Chatbot 🤖")

if "list_of_models" not in st.session_state:
    st.session_state["list_of_models"] = get_list_of_models()

# 选择嵌入模型
selected_embedding_model = st.sidebar.selectbox("Select an embedding model:", st.session_state["list_of_models"])

if st.session_state.get("embedding_model") != selected_embedding_model:
    st.session_state["embedding_model"] = selected_embedding_model

# 选择生成模型
selected_llm_model = st.sidebar.selectbox("Select a LLM model:", st.session_state["list_of_models"])

if st.session_state.get("ollama_model") != selected_llm_model:
    st.session_state["ollama_model"] = selected_llm_model
    st.session_state["llm"] = Ollama(model=selected_llm_model)

folder_path = st.sidebar.text_input("Enter the folder path:", DEFAULT_PATH)

if folder_path:
    if not os.path.isdir(folder_path):
        st.error("The provided path is not a valid directory. Please enter a valid folder path.")
    else:
        if st.sidebar.button("Load FinQA data"):
            if "db" not in st.session_state:
                with st.spinner("Creating embeddings and loading documents into Chroma..."):
                    try:
                        st.session_state["db"] = load_documents_into_database(selected_embedding_model, folder_path, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
                        st.info("Documents loaded into the database successfully!")
                        logging.info("Documents loaded into the database successfully")
                    except Exception as e:
                        st.error(f"Error loading documents: {e}")
                        logging.error(f"Error loading documents: {e}")
else:
    st.warning("Please enter a folder path to load documents into the database.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            stream = get_streaming_chain(prompt, st.session_state.messages, st.session_state["llm"], st.session_state["db"])
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.success("Response generated successfully!")
            logging.info("Response generated successfully")
        except Exception as e:
            st.error(f"Error generating response: {e}")
            logging.error(f"Error generating response: {e}")
