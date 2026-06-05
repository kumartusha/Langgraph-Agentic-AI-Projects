from pydantic import BaseModel, Field
from typing import Dict, List, Any, Callable
from typing_extensions import TypedDict

# ---------------------------------------------------------
# API Request / Response DTOs
# ---------------------------------------------------------

class CodeExecutionRequest(BaseModel):
    code: str = Field(..., description="The raw Python code defining a single function to test")
    function_name: str = Field(..., description="The name of the function to extract and execute")
    arguments: List[Any] = Field(..., description="List of arguments to pass to the function")

class CodeExecutionResponse(BaseModel):
    original_code: str
    final_code: str
    result: Any
    was_patched: bool
    bug_reports: List[str]

# ---------------------------------------------------------
# LangGraph State Definitions
# ---------------------------------------------------------

class HealingAgentState(TypedDict):
    function: Callable
    function_name: str
    function_string: str
    arguments: list
    error: bool
    error_description: str
    new_function_string: str
    bug_report: str
    memory_search_results: list
    memory_ids_to_update: list
    
    # Track metrics for the API response
    all_bug_reports: list
    was_patched: bool
    final_result: Any
