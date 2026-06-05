import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application Settings loaded from environment variables.
    """
    APP_NAME: str = "CRM Call Analyzer API"
    APP_VERSION: str = "1.0.0"
    
    # Try to dynamically locate the root .env file if running in subfolders
    # The user has .env in GenerativeAI_Foundation root.
    # We will let python-dotenv handle it in main.py, so we just expect keys here.

    OPENAI_API_KEY: str = ""
    
    # We can also add other LLM provider keys if needed
    GROQ_API_KEY: str = ""
    GROK_API_KEY: str = "" # Some .env files have this typo

    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
