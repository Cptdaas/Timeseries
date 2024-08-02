import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms.openai import OpenAIChat
from langchain.chains import ConversationalRetrievalChain
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Streamlit app layout
st.set_page_config(page_title="Chatbot with RAG")
st.title("Configure your Broadband Services:")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'retriever' not in st.session_state:
    st.session_state.retriever = None

# Directories
TMP_DIR = Path("C:/prpwork/01AT&TPOC/data")
LOCAL_VECTOR_STORE_DIR = Path("C:/prpwork/01AT&TPOC/data/vector_store")

# Function to load documents
def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
    documents = loader.load()
    return documents

# Function to split documents
def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

# Function to create embeddings and save to local vector store
def embeddings_on_local_vectordb(texts):
    vectordb = Chroma.from_documents(texts, embedding=OpenAIEmbeddings(), persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix())
    vectordb.persist()
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    return retriever

# Function to query the LLM
def query_llm(retriever, query, api_key):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=OpenAIChat(openai_api_key=api_key),
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
    result = result['answer']
    st.session_state.messages.append((query, result))
    return result

# Display document loading section
st.header("Document Loading and Embedding")
api_key = st.text_input("OpenAI API Key", type="password")
if st.button("Load and Process Documents"):
    if api_key:
        documents = load_documents()
        texts = split_documents(documents)
        retriever = embeddings_on_local_vectordb(texts)
        st.session_state.retriever = retriever
        st.success("Documents loaded and processed!")
    else:
        st.warning("Please enter the OpenAI API key.")

# Display query submission section
st.header("Chat with the Bot")
query = st.text_input("Enter your question:")
if st.button("Submit Query"):
    if 'retriever' in st.session_state:
        if api_key:
            answer = query_llm(st.session_state.retriever, query, api_key)
            st.write(f"Answer: {answer}")
        else:
            st.warning("Please enter the OpenAI API key.")
    else:
        st.error("Please load and process documents first.")
import streamlit as st
import random
import time

st.title("Simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
# Display assistant response in chat message container
with st.chat_message("assistant"):
    response = st.write_stream(response_generator())
# Add assistant response to chat history
st.session_state.messages.append({"role": "assistant", "content": response})
# Display chat history
st.header("Chat History")
for i, (question, answer) in enumerate(st.session_state.messages):
    st.write(f"Q{i+1}: {question}")
    st.write(f"A{i+1}: {answer}")
