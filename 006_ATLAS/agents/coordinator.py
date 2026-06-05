"""
agents/coordinator.py
---------------------
Coordinator Agent — the entry-point brain of ATLAS.

Responsibilities:
  1. Analyse the student's current context (courses, tasks, calendar load).
  2. Use a ReACT prompt to decide which specialist agents are needed.
  3. Return a structured coordination plan consumed by the LangGraph router.

Public API:
    coordinator_agent(state)          → LangGraph node callable.
    parse_coordinator_response(resp)  → Parses raw LLM text into a plan dict.
    analyze_context(state)            → Builds a context summary dict.
"""

import json
from typing import Dict

from core.state import AcademicState


# ── Prompt ────────────────────────────────────────────────────────────────────
COORDINATOR_PROMPT = """\
You are a Coordinator Agent using the ReACT framework to orchestrate multiple academic support agents.

AVAILABLE AGENTS:
  • PLANNER    — Handles scheduling and time management
  • NOTEWRITER — Creates study materials and content summaries
  • ADVISOR    — Provides personalised academic guidance

PARALLEL EXECUTION RULES:
  1. Group compatible agents that can run concurrently.
  2. Maintain dependencies between agent executions.
  3. Coordinate results from parallel executions.

REACT PATTERN:
  Thought:     [Analyse request complexity and required support types]
  Action:      [Select optimal agent combination]
  Observation: [Evaluate selected agents' capabilities]
  Decision:    [Finalise agent deployment plan]

ANALYSIS POINTS:
  1. Task Complexity and Scope
  2. Time Constraints
  3. Resource Requirements
  4. Learning Style Alignment
  5. Support Type Needed

CONTEXT:
  Request:         {request}
  Student Context: {context}

FORMAT YOUR RESPONSE AS:
  Thought:     [Analysis of academic needs and context]
  Action:      [Agent selection and grouping strategy]
  Observation: [Expected workflow and dependencies]
  Decision:    [Final agent deployment plan with rationale]
"""


# ── Context Analyser ──────────────────────────────────────────────────────────
async def analyze_context(state: AcademicState) -> Dict:
    """
    Build a structured snapshot of the student's current situation.

    Extracts:
        - Student major, year, and learning style.
        - Which course (if any) the current request relates to.
        - How many calendar events and tasks are active.
        - Study patterns from the profile.

    Args:
        state: Current AcademicState.

    Returns:
        Dict summarising student context for the coordinator prompt.
    """
    profile  = state.get("profile", {})
    calendar = state.get("calendar", {})
    tasks    = state.get("tasks", {})

    courses = profile.get("academic_info", {}).get("current_courses", [])
    request = state["messages"][-1].content.lower()

    # Identify the most relevant course from the request text
    current_course = None
    for course in courses:
        if course["name"].lower() in request:
            current_course = course
            break

    return {
        "student": {
            "major":          profile.get("personal_info", {}).get("major", "Unknown"),
            "year":           profile.get("personal_info", {}).get("academic_year"),
            "learning_style": profile.get("learning_preferences", {}).get("learning_style", {}),
        },
        "course":          current_course,
        "upcoming_events": len(calendar.get("events", [])),
        "active_tasks":    len(tasks.get("tasks", [])),
        "study_patterns":  profile.get("learning_preferences", {}).get("study_patterns", {}),
    }


# ── Response Parser ───────────────────────────────────────────────────────────
def parse_coordinator_response(response: str) -> Dict:
    """
    Parse the LLM's raw ReACT response into a structured coordination plan.

    Strategy:
        1. Start with a safe default (PLANNER only).
        2. Scan for ReACT markers (Thought / Decision).
        3. Add NOTEWRITER if note-taking keywords appear.
        4. Add ADVISOR if guidance keywords appear.
        5. Extract reasoning from the Thought section.

    Args:
        response: Raw LLM text following the ReACT pattern.

    Returns:
        Dict with keys:
            required_agents  — List[str] of agent names to run.
            priority         — Dict[str, int] priority per agent.
            concurrent_groups — List[List[str]] groups to run in parallel.
            reasoning        — str extracted from the Thought section.
    """
    try:
        # Safe default: only PLANNER
        analysis = {
            "required_agents":   ["PLANNER"],
            "priority":          {"PLANNER": 1},
            "concurrent_groups": [["PLANNER"]],
            "reasoning":         response,
        }

        if "Thought:" in response and "Decision:" in response:
            # NOTEWRITER: triggered by note/study-material keywords
            if "NOTEWRITER" in response or "note" in response.lower():
                analysis["required_agents"].append("NOTEWRITER")
                analysis["priority"]["NOTEWRITER"] = 2
                analysis["concurrent_groups"] = [["PLANNER", "NOTEWRITER"]]

            # ADVISOR: triggered by guidance/advice keywords
            if "ADVISOR" in response or "guidance" in response.lower():
                analysis["required_agents"].append("ADVISOR")
                analysis["priority"]["ADVISOR"] = 3

            # Extract reasoning text
            thought_section = response.split("Thought:")[1].split("Action:")[0].strip()
            analysis["reasoning"] = thought_section

        return analysis

    except Exception as exc:
        print(f"⚠️  Coordinator parse error: {exc}")
        return {
            "required_agents":   ["PLANNER"],
            "priority":          {"PLANNER": 1},
            "concurrent_groups": [["PLANNER"]],
            "reasoning":         "Fallback due to parse error.",
        }


# ── LangGraph Node ────────────────────────────────────────────────────────────
async def coordinator_agent(state: AcademicState) -> Dict:
    """
    Primary coordinator node — decides which agents handle the request.

    Flow:
        1. Analyse student context via analyze_context().
        2. Format the COORDINATOR_PROMPT with context + query.
        3. Call the LLM asynchronously via ainvoke().
        4. Parse the response and return the coordination plan.

    Args:
        state: Current AcademicState.

    Returns:
        Dict to merge into state["results"]["coordinator_analysis"].
    """
    from llm.client import get_llm
    llm = get_llm()

    try:
        context = await analyze_context(state)
        query   = state["messages"][-1].content

        # Use ainvoke() for non-blocking async LLM call
        response = (await llm.ainvoke(
            COORDINATOR_PROMPT.format(
                request=query,
                context=json.dumps(context, indent=2),
            )
        )).content

        analysis = parse_coordinator_response(response)

        return {
            "results": {
                "coordinator_analysis": {
                    "required_agents":   analysis.get("required_agents", ["PLANNER"]),
                    "priority":          analysis.get("priority", {"PLANNER": 1}),
                    "concurrent_groups": analysis.get("concurrent_groups", [["PLANNER"]]),
                    "reasoning":         response,
                }
            }
        }

    except Exception as exc:
        print(f"❌ Coordinator error: {exc}")
        return {
            "results": {
                "coordinator_analysis": {
                    "required_agents":   ["PLANNER"],
                    "priority":          {"PLANNER": 1},
                    "concurrent_groups": [["PLANNER"]],
                    "reasoning":         "Error in coordination. Falling back to PLANNER.",
                }
            }
        }
