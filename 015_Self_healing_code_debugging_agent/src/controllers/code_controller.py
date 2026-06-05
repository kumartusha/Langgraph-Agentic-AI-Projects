from fastapi import APIRouter, HTTPException
from src.models.domain_models import CodeExecutionRequest, CodeExecutionResponse
from src.services.self_healing_service import SelfHealingService
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
healing_service = SelfHealingService()

@router.post("/execute", response_model=CodeExecutionResponse)
def execute_code(request: CodeExecutionRequest):
    """
    Submit arbitrary Python code to be executed.
    If it fails, the AI agent will heal it, save the bug pattern to ChromaDB,
    patch the code, and re-execute it.
    """
    try:
        # LangGraph invoke is generally synchronous
        result = healing_service.run_agent(
            code_string=request.code,
            function_name=request.function_name,
            args=request.arguments
        )
        return result
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
