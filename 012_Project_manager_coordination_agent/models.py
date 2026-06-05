"""
Pydantic data models for the Project Manager Assistant.

Defines structured schemas used for LLM structured output and
state management throughout the LangGraph workflow.

Model Hierarchy:
    - Task, TaskList                → Core task definitions
    - TaskDependency, DependencyList → Inter-task dependency mapping
    - TeamMember, Team              → Team composition
    - TaskAllocation, TaskAllocationList → Task-to-member assignments
    - TaskSchedule, Schedule        → Timeline / Gantt scheduling
    - Risk, RiskList                → Per-task risk scoring
    - *Iteration variants          → Multi-iteration tracking wrappers
"""

import uuid
from typing import List
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Core Task Models
# ──────────────────────────────────────────────

class Task(BaseModel):
    """Represents a single actionable project task."""
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the task (auto-generated)",
    )
    task_name: str = Field(description="Name of the task")
    task_description: str = Field(description="Description of the task")
    estimated_days: int = Field(
        description="Estimated number of days to complete the task",
    )


class TaskList(BaseModel):
    """A collection of project tasks."""
    tasks: List[Task] = Field(description="List of tasks")


# ──────────────────────────────────────────────
# Dependency Models
# ──────────────────────────────────────────────

class TaskDependency(BaseModel):
    """Maps a task to its dependent (downstream) tasks using names."""
    task_name: str = Field(description="Name of the task")
    dependent_task_names: List[str] = Field(
        description="List of names of tasks that depend on this task",
    )


class DependencyList(BaseModel):
    """Collection of all task dependency mappings."""
    dependencies: List[TaskDependency] = Field(
        description="List of task dependencies",
    )


# ──────────────────────────────────────────────
# Team Models
# ──────────────────────────────────────────────

class TeamMember(BaseModel):
    """Represents a single team member with their skill profile."""
    name: str = Field(description="Name of the team member")
    profile: str = Field(description="Profile of the team member")


class Team(BaseModel):
    """The full project team roster."""
    team_members: List[TeamMember] = Field(description="List of team members")


# ──────────────────────────────────────────────
# Task Allocation Models
# ──────────────────────────────────────────────

class TaskAllocation(BaseModel):
    """Assignment of a single task to a team member."""
    task: Task = Field(description="Task")
    team_member: TeamMember = Field(
        description="Team members assigned to the task",
    )


class TaskAllocationList(BaseModel):
    """Collection of all task-to-member assignments."""
    task_allocations: List[TaskAllocation] = Field(
        description="List of task allocations",
    )


# ──────────────────────────────────────────────
# Schedule Models
# ──────────────────────────────────────────────

class TaskSchedule(BaseModel):
    """Timeline entry for a single task."""
    task: Task = Field(description="Task")
    start_day: int = Field(description="Start day of the task")
    end_day: int = Field(description="End day of the task")


class Schedule(BaseModel):
    """The complete project schedule (list of task timelines)."""
    schedule: List[TaskSchedule] = Field(
        description="List of task schedules",
    )


# ──────────────────────────────────────────────
# Risk Models
# ──────────────────────────────────────────────

class Risk(BaseModel):
    """Risk assessment for a single task."""
    task: Task = Field(description="Task")
    score: str = Field(description="Risk associated with the task")


class RiskList(BaseModel):
    """Collection of per-task risk scores."""
    risks: List[Risk] = Field(description="List of risks")


# ──────────────────────────────────────────────
# Iteration Tracking Models
# ──────────────────────────────────────────────

class TaskAllocationListIteration(BaseModel):
    """Tracks task allocations across multiple optimization iterations."""
    task_allocations_iteration: List[TaskAllocationList] = Field(
        description="List of task allocations for each iteration",
    )


class ScheduleIteration(BaseModel):
    """Tracks schedules across multiple optimization iterations."""
    schedule: List[Schedule] = Field(
        description="List of task schedules for each iteration",
    )


class RiskListIteration(BaseModel):
    """Tracks risk assessments across multiple optimization iterations."""
    risks_iteration: List[RiskList] = Field(
        description="List of risks for each iteration",
    )
