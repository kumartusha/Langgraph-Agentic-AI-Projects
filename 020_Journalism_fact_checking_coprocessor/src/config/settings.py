from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr
from typing import Optional

class Settings(BaseSettings):
    """
    Application settings for 020_journalism_agent.
    Loads secrets from .env file (up to 3 levels up).
    """
    grok_api_key: SecretStr

    # Traverse up to find .env file
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env", "../../.env", "../../../.env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
