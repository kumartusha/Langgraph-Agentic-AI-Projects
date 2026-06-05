"""
LangGraph workflow construction for the Project Manager Assistant.

Builds the directed state graph, wires nodes and edges,
defines the conditional router, and compiles the final
executable graph with checkpoint memory.

Graph Topology:
    START → task_generation → task_dependencies → task_scheduler
          → task_allocator → risk_assessor
          → [CONDITIONAL] → insight_generator → task_scheduler (loop)
                          → END (if risk improved or max iterations reached)
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
from nodes import (
    task_generation_node,
    task_dependency_node,
    task_scheduler_node,
    task_allocation_node,
    risk_assessment_node,
    insight_generation_node,
)


# ──────────────────────────────────────────────
# Conditional Router
# ──────────────────────────────────────────────

def router(state: AgentState):
    """
    Decide whether to iterate (improve the plan) or terminate.

    Routing Logic:
        1. If max iterations reached → END
        2. If ≥2 iterations completed and latest risk < first risk → END (improved!)
        3. Otherwise → continue to insight_generator for another optimization pass
    """
    max_iteration = state["max_iteration"]
    iteration_number = state["iteration_number"]

    if iteration_number < max_iteration:
        if len(state["project_risk_score_iterations"]) > 1:
            if (
                state["project_risk_score_iterations"][-1]
                < state["project_risk_score_iterations"][0]
            ):
                return END
            else:
                return "insight_generator"
        else:
            return "insight_generator"
    else:
        return END


# ──────────────────────────────────────────────
# Graph Construction
# ──────────────────────────────────────────────

def build_graph():
    """
    Construct and compile the LangGraph workflow.

    Returns:
        A compiled LangGraph instance with checkpoint memory,
        ready for streaming execution.
    """
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("task_generation", task_generation_node)
    workflow.add_node("task_dependencies", task_dependency_node)
    workflow.add_node("task_scheduler", task_scheduler_node)
    workflow.add_node("task_allocator", task_allocation_node)
    workflow.add_node("risk_assessor", risk_assessment_node)
    workflow.add_node("insight_generator", insight_generation_node)

    # Wire sequential edges
    workflow.set_entry_point("task_generation")
    workflow.add_edge("task_generation", "task_dependencies")
    workflow.add_edge("task_dependencies", "task_scheduler")
    workflow.add_edge("task_scheduler", "task_allocator")
    workflow.add_edge("task_allocator", "risk_assessor")

    # Conditional edge: iterate or terminate
    workflow.add_conditional_edges(
        "risk_assessor",
        router,
        ["insight_generator", END],
    )

    # Feedback loop: insights feed back into scheduling
    workflow.add_edge("insight_generator", "task_scheduler")

    # Compile with checkpoint memory for state persistence
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
