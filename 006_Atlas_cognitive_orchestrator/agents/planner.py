"""
agents/planner.py
-----------------
Planner Agent — creates ADHD-aware, energy-optimised study schedules.

Internal subgraph:
    calendar_analyzer → task_analyzer → plan_generator

The subgraph is compiled as a standalone LangGraph and invoked via __call__.
The three node methods are also exposed individually so the main workflow
graph can wire them directly for parallel execution with other agents.

Public API:
    PlannerAgent(llm)
    PlannerAgent.__call__(state) → Dict
    PlannerAgent.calendar_analyzer(state) → Dict
    PlannerAgent.task_analyzer(state) → Dict
    PlannerAgent.plan_generator(state) → Dict
"""

import json
from typing import Dict, List
from datetime import datetime, timezone, timedelta

from langgraph.graph import StateGraph, END, START

from agents.base_agent import ReActAgent
from core.state import AcademicState


# ── Few-shot examples ─────────────────────────────────────────────────────────
_FEW_SHOT_EXAMPLES = [
    {
        "input": "Help with exam prep while managing ADHD and football",
        "thought": "Need to check calendar conflicts and energy patterns",
        "action": "search_calendar",
        "observation": "Football match at 6PM, exam tomorrow 9AM",
        "plan": """\
ADHD-OPTIMISED SCHEDULE:
  PRE-FOOTBALL (2PM–5PM):
    - 3×20-min study sprints
    - Movement breaks
    - Quick reward after each sprint

  FOOTBALL MATCH (6PM–8PM):
    - Use as dopamine reset
    - Formula review during breaks

  POST-MATCH (9PM–12AM):
    - Environment: café noise
    - 15/5 study/break cycles
    - Location changes hourly

  EMERGENCY PROTOCOLS:
    - Focus lost → jumping jacks
    - Overwhelmed  → room change
    - Brain fog    → cold shower""",
    },
    {
        "input": "Struggling with multiple deadlines",
        "thought": "Check task priorities and performance issues",
        "action": "analyze_tasks",
        "observation": "3 assignments due, lowest grade in Calculus",
        "plan": """\
PRIORITY SCHEDULE:
  HIGH-FOCUS SLOTS:
    - Morning: Calculus practice
    - Post-workout: Assignments
    - Night: Quick reviews

  ADHD MANAGEMENT:
    - Task-timer challenges
    - Reward system per completion
    - Study-buddy accountability""",
    },
]


# ── PlannerAgent ──────────────────────────────────────────────────────────────
class PlannerAgent(ReActAgent):
    """
    Specialist agent for creating personalised study schedules.

    Internal workflow:
        1. calendar_analyzer — identify free / busy windows.
        2. task_analyzer     — prioritise pending assignments.
        3. plan_generator    — synthesise a detailed study plan.
    """

    def __init__(self, llm):
        super().__init__(llm)
        self.llm = llm
        self.few_shot_examples = _FEW_SHOT_EXAMPLES
        self.workflow = self._build_subgraph()

    # ── Subgraph ──────────────────────────────────────────────────────────────
    def _build_subgraph(self) -> StateGraph:
        """Wire the internal planner sub-graph and compile it."""
        sg = StateGraph(AcademicState)
        sg.add_node("calendar_analyzer", self.calendar_analyzer)
        sg.add_node("task_analyzer",     self.task_analyzer)
        sg.add_node("plan_generator",    self.plan_generator)

        sg.add_edge("calendar_analyzer", "task_analyzer")
        sg.add_edge("task_analyzer",     "plan_generator")
        sg.set_entry_point("calendar_analyzer")
        return sg.compile()

    # ── Node 1: Calendar Analyser ─────────────────────────────────────────────
    async def calendar_analyzer(self, state: AcademicState) -> Dict:
        """
        Analyse the next 7 days of calendar events.

        Identifies:
            - Available time blocks.
            - Energy impact of each activity.
            - Potential conflicts and recovery periods.

        Returns:
            state update with results["calendar_analysis"].
        """
        events = state["calendar"].get("events", [])
        now    = datetime.now(timezone.utc)
        future = now + timedelta(days=7)

        filtered_events = [
            e for e in events
            if now <= datetime.fromisoformat(e["start"]["dateTime"]) <= future
        ]

        prompt = """\
Analyse these calendar events and identify:

Events: {events}

Focus on:
  - Available time blocks
  - Energy impact of each activity
  - Potential conflicts
  - Recovery periods
  - Study opportunity windows
  - Schedule optimisation suggestions
""".format(events=json.dumps(filtered_events, indent=2))

        response = (await self.llm.ainvoke(prompt)).content

        return {
            "results": {
                "calendar_analysis": {"analysis": response}
            }
        }

    # ── Node 2: Task Analyser ─────────────────────────────────────────────────
    async def task_analyzer(self, state: AcademicState) -> Dict:
        """
        Prioritise and categorise the student's pending tasks.

        Considers:
            - Urgency, complexity, energy requirements.
            - Dependencies, time estimates, learning objectives.

        Returns:
            state update with results["task_analysis"].
        """
        tasks = state["tasks"].get("tasks", [])

        prompt = """\
Analyse these tasks and create a priority structure:

Tasks: {tasks}

Consider:
  - Urgency levels
  - Task complexity
  - Energy requirements
  - Dependencies
  - Required focus levels
  - Time estimations
  - Learning objectives
  - Success criteria
""".format(tasks=json.dumps(tasks, indent=2))

        response = (await self.llm.ainvoke(prompt)).content

        return {
            "results": {
                "task_analysis": {"analysis": response}
            }
        }

    # ── Node 3: Plan Generator ────────────────────────────────────────────────
    async def plan_generator(self, state: AcademicState) -> Dict:
        """
        Synthesise a comprehensive, ADHD-friendly study plan.

        Combines:
            - Profile analysis (learning style).
            - Calendar analysis (free windows).
            - Task analysis (priority order).

        Returns:
            state update with results["final_plan"].
        """
        profile_analysis  = state["results"].get("profile_analysis", {})
        calendar_analysis = state["results"].get("calendar_analysis", {})
        task_analysis     = state["results"].get("task_analysis", {})

        prompt = f"""\
AI Planning Assistant: Create a focused study plan using the ReACT framework.

INPUT CONTEXT:
  Profile Analysis:  {profile_analysis}
  Calendar Analysis: {calendar_analysis}
  Task Analysis:     {task_analysis}

EXAMPLES:
{json.dumps(self.few_shot_examples, indent=2)}

INSTRUCTIONS:
  Follow the ReACT pattern:
    Thought:     Analyse situation and needs
    Action:      Consider all analyses
    Observation: Synthesise findings
    Plan:        Create structured plan

  Address:
    - ADHD management strategies
    - Energy level optimisation
    - Task chunking methods
    - Focus period scheduling
    - Environment switching tactics
    - Recovery period planning
    - Social/sport activity balance

  Include:
    - Emergency protocols
    - Backup strategies
    - Quick wins
    - Reward system
    - Progress tracking
    - Adjustment triggers

  Use informal, encouraging language — you're helping a real student.

FORMAT:
  Thought:     [reasoning and situation analysis]
  Action:      [synthesis approach]
  Observation: [key findings]
  Plan:        [actionable steps and structured schedule]
"""

        response = (await self.llm.ainvoke(prompt)).content

        return {
            "results": {
                "final_plan": {"plan": response}
            }
        }

    # ── Entry Point ───────────────────────────────────────────────────────────
    async def __call__(self, state: AcademicState) -> Dict:
        """Run the full planner subgraph and return the generated plan."""
        try:
            final_state = await self.workflow.ainvoke(state)
            return {"plan": final_state["results"].get("final_plan")}
        except Exception as exc:
            print(f"❌ PlannerAgent error: {exc}")
            return {"plan": "Error generating plan. Please try again."}
