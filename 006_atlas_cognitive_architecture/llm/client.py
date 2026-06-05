"""
llm/client.py
-------------
LLM client setup for ATLAS.

Provides:
  - NeMoLLaMa   : AsyncOpenAI wrapper for NVIDIA NeMo endpoints (legacy).
  - get_llm()   : Factory that returns the currently active LLM (ChatGroq).
"""

import os
from typing import List, Dict, Optional

from openai import AsyncOpenAI
from langchain_groq import ChatGroq

from config.settings import LLMConfig, get_groq_api_key


# ── NVIDIA NeMo LLM (async, legacy) ──────────────────────────────────────────
class NeMoLLaMa:
    """
    Async wrapper for NVIDIA's Nemotron-4-340B model.

    Used via AsyncOpenAI client for non-blocking LLM calls.
    Kept for reference / fallback; active LLM is ChatGroq.
    """

    def __init__(self, api_key: str):
        """
        Args:
            api_key: NVIDIA API authentication key.
        """
        self.config = LLMConfig()
        self.client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=api_key,
        )
        self._is_authenticated = False

    async def check_auth(self) -> bool:
        """
        Verify API authentication with a lightweight test request.

        Returns:
            bool: True if authenticated, False otherwise.
        """
        try:
            await self.agenerate([{"role": "user", "content": "test"}], temperature=0.1)
            self._is_authenticated = True
            return True
        except Exception as exc:
            print(f"❌ Authentication failed: {exc}")
            return False

    async def agenerate(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a text response from the NeMo model.

        Args:
            messages:    List of {"role": ..., "content": ...} dicts.
            temperature: Sampling temperature (0.0–1.0). Defaults to config value.

        Returns:
            str: Model response text.
        """
        config = LLMConfig()
        completion = await self.client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=temperature or config.default_temp,
            max_tokens=config.max_tokens,
            stream=False,
        )
        return completion.choices[0].message.content


# ── Active LLM factory ────────────────────────────────────────────────────────
def get_llm() -> ChatGroq:
    """
    Return the currently active LLM instance (ChatGroq / LLaMA-3.3-70B).

    This is the single place to swap the LLM provider for the whole system.

    Returns:
        ChatGroq: Configured language model instance.
    """
    config = LLMConfig()
    return ChatGroq(
        model=config.groq_model,
        api_key=get_groq_api_key(),
    )
