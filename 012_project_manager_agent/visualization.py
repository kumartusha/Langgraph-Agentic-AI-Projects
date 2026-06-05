"""
Visualization utilities for the Project Manager Assistant.

Generates a Plotly Gantt chart from the final project plan,
mapping tasks to team members with color-coded timelines.
"""

import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta


def build_gantt_chart(final_state: dict, title: str = "Project Plan - Gantt Chart"):
    """
    Build and display a Gantt chart from the final workflow state.

    Args:
        final_state: The completed AgentState dictionary from the LangGraph run.
        title:       Chart title (defaults to 'Project Plan - Gantt Chart').

    Returns:
        A Plotly Figure object (also calls fig.show() for display).
    """
    # Extract schedule and allocation data
    task_schedules = final_state["schedule"].schedule
    task_allocations = final_state["task_allocations"].task_allocations

    # Build schedule DataFrame
    schedule_rows = []
    for ts in task_schedules:
        schedule_rows.append([
            ts.task.task_name,
            ts.start_day,
            ts.end_day,
        ])
    df_schedule = pd.DataFrame(
        schedule_rows,
        columns=["task_name", "start", "end"],
    )

    # Build allocation DataFrame
    allocation_rows = []
    for ta in task_allocations:
        allocation_rows.append([
            ta.task.task_name,
            ta.team_member.name,
        ])
    df_allocation = pd.DataFrame(
        allocation_rows,
        columns=["task_name", "team_member"],
    )

    # Merge schedule + allocation
    df = df_allocation.merge(df_schedule, on="task_name")

    # Convert day offsets to actual dates
    current_date = datetime.today()
    df["start"] = df["start"].apply(lambda x: current_date + timedelta(days=x))
    df["end"] = df["end"].apply(lambda x: current_date + timedelta(days=x))

    # Rename for display
    df.rename(columns={"team_member": "Team Member"}, inplace=True)
    df.sort_values(by="Team Member", inplace=True)

    # Create Gantt chart
    fig = px.timeline(
        df,
        x_start="start",
        x_end="end",
        y="task_name",
        color="Team Member",
        title=title,
    )

    # Polish layout
    fig.update_layout(
        xaxis_title="Timeline",
        yaxis_title="Tasks",
        yaxis=dict(autorange="reversed"),
        title_x=0.5,
    )

    fig.show()
    return fig
