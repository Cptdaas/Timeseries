from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms.openai import OpenAIChat
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter

import os

app = FastAPI()

# Directory settings
TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

# Session state simulation
session_state = {'messages': [], 'retriever': None}

class QueryModel(BaseModel):
    query: str

class APIKeyModel(BaseModel):
    openai_api_key: str
    pinecone_api_key: str
    pinecone_env: str
    pinecone_index: str

def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

def embeddings_on_local_vectordb(texts):
    vectordb = Chroma.from_documents(texts, embedding=OpenAIEmbeddings(),
                                     persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix())
    vectordb.persist()
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    return retriever

def query_llm(retriever, query, api_key):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=OpenAIChat(openai_api_key=api_key),
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa_chain({'question': query, 'chat_history': session_state['messages']})
    result = result['answer']
    session_state['messages'].append((query, result))
    return result

@app.post("/load_documents")
async def load_and_process_documents(api_key_model: APIKeyModel):
    global session_state
    try:
        documents = load_documents()
        texts = split_documents(documents)
        retriever = embeddings_on_local_vectordb(texts)
        session_state['retriever'] = retriever
        return JSONResponse(content={"message": "Documents loaded and processed!"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def submit_query(query_model: QueryModel, request: Request):
    global session_state
    api_key = request.headers.get("X-API-KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key not provided in the request headers.")
    
    if session_state['retriever'] is None:
        raise HTTPException(status_code=400, detail="Please load and process documents first.")
    
    try:
        answer = query_llm(session_state['retriever'], query_model.query, api_key)
        return JSONResponse(content={"answer": answer}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat_history")
async def get_chat_history():
    global session_state
    chat_history = [{"question": q, "answer": a} for q, a in session_state['messages']]
    return JSONResponse(content={"chat_history": chat_history}, status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
