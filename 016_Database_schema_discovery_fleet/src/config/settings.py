import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv(dotenv_path="../../../../.env")

class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "dummy_key")
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "sqlite:///data/chinook.db")

settings = Settings()
