"""
agents/base_agent.py
--------------------
Base class for all ReACT agents in ATLAS.

Provides:
  - ReActAgent  : Base class with shared tools (calendar search, task analysis, etc.)
  - AgentAction : Pydantic model for a single agent action decision.
  - AgentOutput : Pydantic model for an agent's action result.
"""

from typing import List, Dict, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from core.state import AcademicState


# ── Pydantic Models ───────────────────────────────────────────────────────────
class AgentAction(BaseModel):
    """
    Represents one action decision made by a ReACT agent.

    Attributes:
        action:       The action to perform (e.g. "search_calendar").
        thought:      The reasoning behind choosing this action.
        tool:         Optional specific tool name to invoke.
        action_input: Optional parameters to pass to the tool.

    Example:
        >>> AgentAction(
        ...     action="search_calendar",
        ...     thought="Need to check for scheduling conflicts",
        ...     tool="calendar_search",
        ...     action_input={"date_range": "next_week"},
        ... )
    """
    action: str
    thought: str
    tool: Optional[str] = None
    action_input: Optional[Dict] = None


class AgentOutput(BaseModel):
    """
    Represents the result returned after executing an agent action.

    Attributes:
        observation: Human-readable description of the result.
        output:      Structured data produced by the action.

    Example:
        >>> AgentOutput(
        ...     observation="Found 3 free time slots next week",
        ...     output={"free_slots": ["Mon 2PM", "Wed 10AM", "Fri 3PM"]},
        ... )
    """
    observation: str
    output: Dict


# ── Base ReACT Agent ──────────────────────────────────────────────────────────
class ReActAgent:
    """
    Base class for all ATLAS agents using the ReACT pattern.

    ReACT = Reason → Act → Observe (loop until done).

    Shared tools available to all child agents:
        - search_calendar      : Filter upcoming calendar events.
        - analyze_tasks        : Return active task list from state.
        - check_learning_style : Extract learning style from student profile.
        - check_performance    : Extract course performance from profile.

    Child agents extend this class and add domain-specific tools
    or subgraph workflows (see PlannerAgent, NoteWriterAgent, AdvisorAgent).
    """

    def __init__(self, llm):
        """
        Args:
            llm: Language model instance shared across all agents.
        """
        self.llm = llm
        self.few_shot_examples: List[Dict] = []

        # Register all shared tools
        self.tools = {
            "search_calendar":     self.search_calendar,
            "analyze_tasks":       self.analyze_tasks,
            "check_learning_style": self.check_learning_style,
            "check_performance":   self.check_performance,
        }

    # ── Shared Tools ──────────────────────────────────────────────────────────
    async def search_calendar(self, state: AcademicState) -> List[Dict]:
        """Return all future calendar events from the state."""
        events = state["calendar"].get("events", [])
        now = datetime.now(timezone.utc)
        return [
            e for e in events
            if datetime.fromisoformat(e["start"]["dateTime"]) > now
        ]

    async def analyze_tasks(self, state: AcademicState) -> List[Dict]:
        """Return all tasks from the state."""
        return state["tasks"].get("tasks", [])

    async def check_learning_style(self, state: AcademicState) -> AcademicState:
        """
        Extract and store learning style preferences from the student profile.

        Updates:
            state["results"]["learning_analysis"]
        """
        profile = state["profile"]
        learning_data = {
            "style":    profile.get("learning_preferences", {}).get("learning_style", {}),
            "patterns": profile.get("learning_preferences", {}).get("study_patterns", {}),
        }
        state.setdefault("results", {})["learning_analysis"] = learning_data
        return state

    async def check_performance(self, state: AcademicState) -> AcademicState:
        """
        Extract and store current course performance from the student profile.

        Updates:
            state["results"]["performance_analysis"]
        """
        profile = state["profile"]
        courses = profile.get("academic_info", {}).get("current_courses", [])
        state.setdefault("results", {})["performance_analysis"] = {"courses": courses}
        return state
