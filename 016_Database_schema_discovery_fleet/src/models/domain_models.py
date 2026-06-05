from pydantic import BaseModel
from typing import Annotated, TypedDict, List, Optional
from typing_extensions import NotRequired
import networkx as nx

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    response: str
    db_results: Optional[str] = None
    plan: Optional[List[str]] = None

def db_graph_reducer(previous_value: Optional[nx.Graph], new_value: nx.Graph) -> nx.Graph:
    if previous_value is None:
        return new_value
    return previous_value

def plan_reducer(previous_value: Optional[List[str]], new_value: List[str]) -> List[str]:
    return new_value if new_value is not None else previous_value

def classify_input_reducer(previous_value: Optional[str], new_value: str) -> str:
    return new_value

class ConversationState(TypedDict):
    question: str
    input_type: Annotated[str, classify_input_reducer]
    plan: Annotated[List[str], plan_reducer]
    db_results: NotRequired[str]
    response: NotRequired[str]
    db_graph: Annotated[Optional[nx.Graph], db_graph_reducer]
