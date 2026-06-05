"""
core/state.py
-------------
Defines the shared AcademicState that flows through every LangGraph node.
Also provides the dict_reducer used for recursive state merging.
"""

from typing import Annotated, List, Dict, Any, TypeVar
from langchain_core.messages import BaseMessage
from operator import add


T = TypeVar('T')


# ── State Reducer ─────────────────────────────────────────────────────────────
def dict_reducer(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two dictionaries recursively (deep merge).

    Example:
        dict1 = {"a": {"x": 1}, "b": 2}
        dict2 = {"a": {"y": 2}, "c": 3}
        result = {"a": {"x": 1, "y": 2}, "b": 2, "c": 3}
    """
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = dict_reducer(merged[key], value)
        else:
            merged[key] = value
    return merged


# ── Master State ──────────────────────────────────────────────────────────────
class AcademicState(dict):
    """
    Master state container for the entire ATLAS workflow.

    Fields:
        messages  — Conversation history; appended on each update.
        profile   — Student profile data (deep-merged).
        calendar  — Scheduled calendar events (deep-merged).
        tasks     — Active tasks / assignments (deep-merged).
        results   — Intermediate & final agent outputs (deep-merged).
    """
    messages: Annotated[List[BaseMessage], add]
    profile:  Annotated[Dict, dict_reducer]
    calendar: Annotated[Dict, dict_reducer]
    tasks:    Annotated[Dict, dict_reducer]
    results:  Annotated[Dict[str, Any], dict_reducer]


# Keep TypedDict variant for LangGraph's type annotations
from typing import TypedDict

class AcademicState(TypedDict):  # noqa: F811
    messages: Annotated[List[BaseMessage], add]
    profile:  Annotated[Dict, dict_reducer]
    calendar: Annotated[Dict, dict_reducer]
    tasks:    Annotated[Dict, dict_reducer]
    results:  Annotated[Dict[str, Any], dict_reducer]
