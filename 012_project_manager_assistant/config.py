"""
Configuration module for the Project Manager Assistant.

Handles environment variable loading and LLM instantiation.
Supports multiple LLM providers: Groq, Azure OpenAI, and OpenAI.
"""

import os
from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq

# Dynamically find the .env file in the GenerativeAI_Foundation folder (3 levels up)
base_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.abspath(os.path.join(base_dir, "..", "..", "..", ".env"))
load_dotenv(dotenv_path=env_path)


# def get_llm(provider: str = "groq"):
#     """
#     Factory function to instantiate the LLM based on the selected provider.

#     Args:
#         provider: The LLM provider to use. Options: 'groq', 'azure', 'openai'.
#                   Defaults to 'groq'.

#     Returns:
#         A LangChain chat model instance.

#     Raises:
#         ValueError: If an unsupported provider is specified.

#     Environment Variables Required:
#         - For 'groq':   GROK_API_KEY
#         - For 'azure':  AZURE_OPENAI_API_KEY, OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT
#         - For 'openai': OPENAI_API_KEY
#     """
#     # if provider == "groq":
#     #     return ChatGroq(
#     #         model="qwen/qwen3-32b",
#     #         api_key=os.getenv("GROK_API_KEY"),
#     #     )
#     # elif provider == "azure":
#     #     from langchain_openai import AzureChatOpenAI
#     #     return AzureChatOpenAI(deployment_name="gpt-4o-mini")
#     # elif provider == "openai":
#     #     from langchain_openai import ChatOpenAI
#     #     return ChatOpenAI(model="gpt-4o-mini")
#     # else:
#     #     raise ValueError(
#     #         f"Unsupported LLM provider: '{provider}'. "
#     #         "Choose from 'groq', 'azure', or 'openai'."
#     #     )
#     if provider == "groq":
#         return ChatGroq(model="qwen/qwen3-32b", api_key=os.getenv("GROK_API_KEY"))
#     else:
#         raise ValueError(
#             f"Unsupported LLM provider: '{provider}'. "
#             "Choose from 'groq', 'azure', or 'openai'."
#         )

# Disable LangSmith tracing to suppress 403 Forbidden warnings
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Default LLM instance used across the application
# llm = get_llm("groq")

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=os.getenv("GROK_API_KEY") or os.getenv("GROQ_API_KEY"),
    timeout=120,
    temperature=0.7,
    max_tokens=4096,
    max_retries=3,
)