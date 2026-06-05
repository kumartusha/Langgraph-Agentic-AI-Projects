import os
import uvicorn
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Set up logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Load Environment Variables BEFORE importing Settings
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
root_env_path = os.path.abspath(os.path.join(current_dir, "../../../.env"))

if os.path.exists(root_env_path):
    load_dotenv(dotenv_path=root_env_path, override=True)
    logger.info(f"Loaded environment variables from {root_env_path}")
else:
    logger.warning(f"Root .env file not found at {root_env_path}")

# Now we can safely import settings, routes, and middleware
from src.config.settings import settings
from src.routes.api import api_router
from src.middleware.error_handler import global_exception_handler

# ---------------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------------
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="A LangGraph agent that dynamically executes, heals, and patches code."
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Error Handler
app.add_exception_handler(Exception, global_exception_handler)

# Include Routers
app.include_router(api_router, prefix="/api/v1")

@app.get("/health", tags=["Health"])
def health_check():
    """Basic health check endpoint"""
    return {"status": "ok", "app": settings.APP_NAME}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
