"""
graph/workflow.py
-----------------
Main LangGraph StateGraph for the ATLAS multi-agent system.

Wires together ALL agent nodes into a single compiled workflow:

    START
      └─► coordinator
            └─► profile_analyzer
                  ├─► calendar_analyzer ─► task_analyzer ─► plan_generator ─┐
                  ├─► notewriter_analyze ─► notewriter_generate ─────────────┤
                  └─► advisor_analyze ─► advisor_generate ───────────────────┘
                                                                              └─► execute ─► END
                                                                                    │
                                                                               (loop back if
                                                                                incomplete)

Public API:
    create_agents_graph(llm) → CompiledStateGraph
"""

from typing import List, Union, Literal

from langgraph.graph import StateGraph, END, START

from core.state import AcademicState
from agents.coordinator import coordinator_agent
from agents.profile_analyzer import profile_analyzer
from agents.planner import PlannerAgent
from agents.note_writer import NoteWriterAgent
from agents.advisor import AdvisorAgent
from executor.agent_executor import AgentExecutor


def create_agents_graph(llm):
    """
    Build and compile the full ATLAS multi-agent LangGraph workflow.

    Node responsibilities:
        coordinator       — Decides which agents are needed (ReACT).
        profile_analyzer  — Extracts learning patterns from student profile.
        calendar_analyzer — Planner: identifies free/busy windows.
        task_analyzer     — Planner: prioritises assignments.
        plan_generator    — Planner: synthesises final study plan.
        notewriter_analyze — NoteWriter: determines note structure.
        notewriter_generate — NoteWriter: generates study notes.
        advisor_analyze   — Advisor: understands student's situation.
        advisor_generate  — Advisor: creates personalised guidance.
        execute           — AgentExecutor: concurrent orchestration + fallback.

    Routing logic:
        profile_analyzer → conditional → [calendar_analyzer |
                                           notewriter_analyze |
                                           advisor_analyze]
        execute → conditional → coordinator (if incomplete) | END

    Args:
        llm: Shared language model instance.

    Returns:
        Compiled LangGraph StateGraph.
    """
    # ── Agent instances ────────────────────────────────────────────────────────
    planner_agent    = PlannerAgent(llm)
    notewriter_agent = NoteWriterAgent(llm)
    advisor_agent    = AdvisorAgent(llm)
    executor         = AgentExecutor(llm)

    # ── Graph ──────────────────────────────────────────────────────────────────
    workflow = StateGraph(AcademicState)

    # Top-level coordination nodes
    workflow.add_node("coordinator",      coordinator_agent)
    workflow.add_node("profile_analyzer", profile_analyzer)
    workflow.add_node("execute",          executor.execute)

    # Planner sub-nodes (exposed individually for parallel wiring)
    workflow.add_node("calendar_analyzer", planner_agent.calendar_analyzer)
    workflow.add_node("task_analyzer",     planner_agent.task_analyzer)
    workflow.add_node("plan_generator",    planner_agent.plan_generator)

    # NoteWriter sub-nodes
    workflow.add_node("notewriter_analyze",  notewriter_agent.analyze_learning_style)
    workflow.add_node("notewriter_generate", notewriter_agent.generate_notes)

    # Advisor sub-nodes
    workflow.add_node("advisor_analyze",  advisor_agent.analyze_situation)
    workflow.add_node("advisor_generate", advisor_agent.generate_guidance)

    # ── Edges: main flow ───────────────────────────────────────────────────────
    workflow.add_edge(START, "coordinator")
    workflow.add_edge("coordinator", "profile_analyzer")

    # ── Conditional routing: profile_analyzer → agent entry points ─────────────
    def route_to_parallel_agents(state: AcademicState) -> List[str]:
        """
        Route to the correct agent entry points based on coordinator analysis.

        Returns a list of node names — LangGraph will run them in parallel.
        Defaults to calendar_analyzer (Planner) if no agents are specified.
        """
        analysis        = state["results"].get("coordinator_analysis", {})
        required_agents = analysis.get("required_agents", [])
        next_nodes: List[str] = []

        if "PLANNER"    in required_agents:
            next_nodes.append("calendar_analyzer")
        if "NOTEWRITER" in required_agents:
            next_nodes.append("notewriter_analyze")
        if "ADVISOR"    in required_agents:
            next_nodes.append("advisor_analyze")

        return next_nodes if next_nodes else ["calendar_analyzer"]

    workflow.add_conditional_edges(
        "profile_analyzer",
        route_to_parallel_agents,
        ["calendar_analyzer", "notewriter_analyze", "advisor_analyze"],
    )

    # ── Edges: Planner internal flow ──────────────────────────────────────────
    workflow.add_edge("calendar_analyzer", "task_analyzer")
    workflow.add_edge("task_analyzer",     "plan_generator")
    workflow.add_edge("plan_generator",    "execute")

    # ── Edges: NoteWriter internal flow ───────────────────────────────────────
    workflow.add_edge("notewriter_analyze",  "notewriter_generate")
    workflow.add_edge("notewriter_generate", "execute")

    # ── Edges: Advisor internal flow ──────────────────────────────────────────
    workflow.add_edge("advisor_analyze",  "advisor_generate")
    workflow.add_edge("advisor_generate", "execute")

    # ── Completion check: loop or END ─────────────────────────────────────────
    def should_end(state: AcademicState) -> Union[Literal["coordinator"], Literal["end"]]:
        """
        Determine whether all required agents have completed.

        If the set of completed agent outputs is a superset of required agents
        → return END. Otherwise loop back to coordinator.
        """
        analysis = state["results"].get("coordinator_analysis", {})
        executed = set(state["results"].get("agent_outputs", {}).keys())
        required = set(a.lower() for a in analysis.get("required_agents", []))
        return END if required.issubset(executed) else "coordinator"

    workflow.add_conditional_edges(
        "execute",
        should_end,
        {
            "coordinator": "coordinator",
            END:           END,
        },
    )

    return workflow.compile()
