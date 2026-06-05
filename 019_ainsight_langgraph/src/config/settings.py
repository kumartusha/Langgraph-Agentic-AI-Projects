from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings and environment variables.
    """
    grok_api_key: str = ""
    tavily_api_key: str = ""
    
    # Model config
    groq_model: str = "llama-3.3-70b-versatile"
    
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env", "../../.env", "../../../.env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
