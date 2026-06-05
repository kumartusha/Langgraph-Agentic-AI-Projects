import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load env variables from root folder before importing settings
# This ensures OPENAI_API_KEY is available if running locally
current_dir = os.path.dirname(os.path.abspath(__file__))
root_env = os.path.join(current_dir, "../../../.env")
load_dotenv(root_env)

from src.config.settings import settings
from src.routes.api import api_router
from src.middleware.error_handler import global_exception_handler

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="An AI-powered monolithic REST API for transcribing and analyzing sales calls."
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception Handlers
app.add_exception_handler(Exception, global_exception_handler)

# Include Routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": settings.APP_VERSION}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)



# How to run this app.abs

# cd /Users/apple/Desktop/DesktopBackup/GenerativeAI_Foundation/Langgraph/projects/013_CRM_Call_analyzer
# python3 -m uvicorn main:app --reload

# 1. Open the Interactive Dashboard
# Open your web browser (Chrome, Safari, etc.) and go to this exact URL: 👉 http://127.0.0.1:8001/docs

# 2. Test the Audio Upload Endpoint
# Once that page loads, you will see a beautiful interface displaying your API endpoints.

# Click on the green row that says POST /api/v1/analyze/audio.
# Click the "Try it out" button on the right side.
# You will see a file upload button. Click "Choose File" and select the dog.mp3 file (or any other short audio file you have).
# Click the large blue "Execute" button at the bottom.
# 3. See the Result
# The API will take a few moments to transcribe the audio and run the CrewAI analysis. Once finished, scroll down slightly to the 
# "Server response" section. You will see a perfectly formatted JSON block containing your actionable insights, sentiment analysis, key phrases, and agent score!