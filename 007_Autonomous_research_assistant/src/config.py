import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Setup standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# API Configurations
CORE_API_KEY = os.environ.get("CORE_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

def get_logger(name: str) -> logging.Logger:
    """Returns a configured logger instance."""
    logger = logging.getLogger(name)
    return logger

logger = get_logger(__name__)

if not CORE_API_KEY:
    logger.warning("CORE_API_KEY is not set in environment. Tools relying on it may fail.")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY is not set in environment. LLM calls will fail.")
