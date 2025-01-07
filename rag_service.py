import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import openai

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def get_video_id(url):
    """Extract the video ID from a YouTube URL."""
    import re
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None

def fetch_transcript(video_url):
    """Fetch the transcript of a YouTube video."""
    video_id = get_video_id(video_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")
    
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([entry['text'] for entry in transcript])

def create_retriever(text):
    """Split text, embed it, and create a retriever."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore.as_retriever()

def summarize_video_with_rag(video_url):
    """Fetch and summarize a YouTube video using RAG."""
    try:
        # Step 1: Fetch transcript
        transcript = fetch_transcript(video_url)
        
        # Step 2: Create retriever
        retriever = create_retriever(transcript)
        
        # Step 3: Use RetrievalQA for summarization
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",  # Use a chat model like gpt-4 or gpt-3.5-turbo
            temperature=0,
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
        )
        
        # Step 4: Generate summary
        query = f"""I have the transcript of a YouTube video, and I need a detailed summary:
                    Requirements:
                        1.	Summarize the key points from the transcript.
                        2.	Provide brief explanations for each key point.
                        3.	Maintain the sequence of ideas as they appear in the transcript.
                        4.	Use a clear and concise tone.

                    Transcript: {transcript}

                    Generate a detailed summary based on the provided transcript."""
        result = qa_chain.invoke(query)
        return result
    except Exception as e:
        raise RuntimeError(f"Error during summarization: {e}")