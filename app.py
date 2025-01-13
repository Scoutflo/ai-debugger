from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from tempfile import NamedTemporaryFile
from typing import Dict
import os
from dotenv import load_dotenv
import tiktoken
from langchain_community.document_loaders import PyPDFLoader


# Preload the cl100k_base tokenizer
tiktoken.get_encoding("cl100k_base")

load_dotenv()

app = FastAPI()

# Add CORS middleware if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store session data
pdf_sessions: Dict[str, Dict] = {}

# API Key for OpenAI
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF and initialize a chat session.
    """
    try:
        # Save uploaded file to a temporary location
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name
        
        # Load and parse the PDF
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        # Generate embeddings and store in FAISS
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_documents(chunks, embeddings)

        # Store session data
        session_id = file.filename  # Use the filename as the session ID
        pdf_sessions[session_id] = {"vector_store": vector_store}

        return JSONResponse(content={"message": "PDF uploaded successfully", "session_id": session_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/chat/")
async def chat(session_id: str = Form(...), query: str = Form(...)):
    """
    Endpoint to chat with a previously uploaded PDF session.
    """
    if session_id not in pdf_sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a PDF first.")

    # Retrieve the vector store and LLM
    session_data = pdf_sessions[session_id]
    vector_store = session_data["vector_store"]
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Generate an answer to the query
    try:
        answer = qa_chain.run(query)
        return JSONResponse(content={"query": query, "answer": answer})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during chat: {str(e)}")


@app.get("/")
def root():
    return {"message": "Welcome to the PDF Chat API!"}