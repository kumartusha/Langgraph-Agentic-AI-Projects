"""
Agent State definition for the Project Manager Assistant.

The AgentState is a TypedDict that flows through every node in the
LangGraph workflow. It serves as the single source of truth for
the entire planning pipeline.
"""

from typing import List, TypedDict

from models import (
    DependencyList,
    RiskList,
    RiskListIteration,
    Schedule,
    TaskAllocationList,
    TaskList,
    Team,
)


class AgentState(TypedDict):
    """
    The project manager agent state.

    This state is shared across all nodes in the LangGraph workflow.
    Each node reads from and writes to this state to pass data
    between pipeline stages.

    Attributes:
        project_description:            Raw project description text.
        team:                           Team roster with skill profiles.
        tasks:                          Generated task breakdown.
        dependencies:                   Inter-task dependency map.
        schedule:                       Current iteration's schedule.
        task_allocations:               Current iteration's task assignments.
        risks:                          Current iteration's risk scores.
        iteration_number:               Current optimization iteration index.
        max_iteration:                  Maximum allowed iterations.
        insights:                       LLM-generated improvement insights.
        schedule_iteration:             History of schedules across iterations.
        task_allocations_iteration:     History of allocations across iterations.
        risks_iteration:               History of risk assessments across iterations.
        project_risk_score_iterations:  History of total risk scores per iteration.
    """
    project_description: str
    team: Team
    tasks: TaskList
    dependencies: DependencyList
    schedule: Schedule
    task_allocations: TaskAllocationList
    risks: RiskList
    iteration_number: int
    max_iteration: int
    insights: List[str]
    schedule_iteration: List[Schedule]
    task_allocations_iteration: List[TaskAllocationList]
    risks_iteration: List[RiskListIteration]
    project_risk_score_iterations: List[int]
