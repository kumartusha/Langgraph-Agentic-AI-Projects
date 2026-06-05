import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application Settings loaded from environment variables.
    The root .env file will be dynamically loaded in main.py
    """
    APP_NAME: str = "Self Healing Code API"
    APP_VERSION: str = "1.0.0"
    
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY") or "dummy_key"
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY") or "dummy_key"

    # LLM Settings (We'll use Groq for speed by default, but fallback to OpenAI if wanted)
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    
    # Vector DB Settings
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma"
    CHROMA_COLLECTION_NAME: str = "bug-reports"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
