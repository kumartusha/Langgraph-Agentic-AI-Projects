"""
config/settings.py
------------------
Central configuration for ATLAS.
Handles environment loading and LLM hyperparameters.
"""

import os
from dotenv import load_dotenv


# ── Load .env from project root ──────────────────────────────────────────────
load_dotenv()


# ── LLM Configuration ─────────────────────────────────────────────────────────
class LLMConfig:
    """Hyperparameters and endpoint settings for the language model."""
    # NVIDIA NeMo endpoint (legacy / alternative LLM)
    base_url: str = "https://integrate.api.nvidia.com/v1"
    model: str = "nvidia/nemotron-4-340b-instruct"
    max_tokens: int = 1024
    default_temp: float = 0.5

    # Groq model (currently active)
    groq_model: str = "llama-3.3-70b-versatile"


# ── API Key helpers ───────────────────────────────────────────────────────────
def get_groq_api_key() -> str:
    """Return the Groq API key from environment."""
    key = os.getenv("GROK_API_KEY")
    if not key:
        raise EnvironmentError("GROK_API_KEY not found. Add it to your .env file.")
    return key


def get_nemotron_api_key() -> str:
    """Return the NVIDIA Nemotron API key from environment (optional)."""
    return os.getenv("NEMOTRON_3_NANO_OMNI_30B_A3B_REASONING_KEY", "")


def configure_api_keys() -> bool:
    """
    Validate that at least one LLM API key is present.

    Returns:
        bool: True if a key is found, False otherwise.
    """
    groq_key = os.getenv("GROK_API_KEY")
    nemotron_key = os.getenv("NEMOTRON_3_NANO_OMNI_30B_A3B_REASONING_KEY")

    if groq_key:
        print("✅ GROK_API_KEY configured.")
        return True
    if nemotron_key:
        print("✅ NEMOTRON key configured.")
        return True

    print("❌ No API keys found. Please add GROK_API_KEY to your .env file.")
    return False
