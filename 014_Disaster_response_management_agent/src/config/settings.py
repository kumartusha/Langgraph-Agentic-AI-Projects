import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Weather Disaster Management API"
    VERSION: str = "1.0.0"
    
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY") or "dummy_key"
    WEATHER_API_KEY: str = os.getenv("W_API_KEY") or "dummy_key"

    # LLM Settings
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    
    # Mock Email settings
    SENDER_EMAIL: str = "system@weatherdisaster.ai"
    RECEIVER_EMAIL: str = "admin@weatherdisaster.ai"

settings = Settings()
