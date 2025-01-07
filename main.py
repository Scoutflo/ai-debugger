from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_service import summarize_video_with_rag

app = FastAPI()

class SummarizeRequest(BaseModel):
    url: str

# API Endpoint
@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    youtube_url = request.url
    
    if not youtube_url:
        raise HTTPException(status_code=400, detail="YouTube URL is required")
    
    try:
        summary = summarize_video_with_rag(youtube_url)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))