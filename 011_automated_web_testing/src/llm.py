import os
from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq

load_dotenv()

if "GROK_API_KEY" in os.environ:
    os.environ["GROK_API_KEY"] = os.environ.get("GROK_API_KEY")

# Initialize the LLM instance
llm = ChatGroq(api_key=os.getenv("GROK_API_KEY"), model="qwen/qwen3-32b", temperature=0)
