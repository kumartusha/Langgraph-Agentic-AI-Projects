from fastapi import FastAPI
import uvicorn
import logging
import os
from dotenv import load_dotenv

# Load env variables from root folder before importing settings
current_dir = os.path.dirname(os.path.abspath(__file__))
root_env = os.path.join(current_dir, "../../../.env")
load_dotenv(root_env)

from src.routes.api import api_router
from src.middleware.error_handler import global_exception_handler
from src.config.settings import settings

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Monolithic FastAPI for LangGraph-based Weather Disaster Management."
)

# Exception Handlers
app.add_exception_handler(Exception, global_exception_handler)

# Include Routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "version": settings.VERSION}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
