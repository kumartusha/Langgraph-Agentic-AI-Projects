"""
Entry point for the Project Manager Assistant.

Orchestrates the full pipeline:
    1. Load project description and team data
    2. Build the LangGraph workflow
    3. Stream execution through all nodes
    4. Visualize the final plan as a Gantt chart

Usage:
    python main.py
"""

import os
import pandas as pd

from models import Team, TeamMember
from graph import build_graph
from visualization import build_gantt_chart


# ──────────────────────────────────────────────
# Data Loading Utilities
# ──────────────────────────────────────────────

def get_project_description(file_path: str) -> str:
    """Read the project description from a text file."""
    with open(file_path, "r") as file:
        content = file.read()
    return content


def get_team(file_path: str) -> Team:
    """
    Read team members from a CSV file.

    Expected CSV columns: Name, Profile Description
    """
    team_df = pd.read_csv(file_path)
    team_members = [
        TeamMember(name=row["Name"], profile=row["Profile Description"])
        for _, row in team_df.iterrows()
    ]
    return Team(team_members=team_members)


# ──────────────────────────────────────────────
# Main Execution
# ──────────────────────────────────────────────

def main():
    """Run the Project Manager Assistant pipeline."""

    # Resolve data file paths relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_desc_path = os.path.join(base_dir, "data", "project_description.txt")
    team_csv_path = os.path.join(base_dir, "data", "team.csv")

    # Load inputs
    project_description = get_project_description(project_desc_path)
    team = get_team(team_csv_path)

    # ANSI Color Codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"

    print(f"\n{CYAN}{BOLD}📋 Project:{RESET} {project_description.strip()}")
    print(f"{CYAN}{BOLD}👥 Team:{RESET} {', '.join(m.name for m in team.team_members)}")
    print(f"{BLUE}─{RESET}" * 60)

    # Build the graph
    graph_plan = build_graph()

    # Initialize agent state
    state_input = {
        "project_description": project_description,
        "team": team,
        "insights": "",
        "iteration_number": 0,
        "max_iteration": 2,
        "schedule_iteration": [],
        "task_allocations_iteration": [],
        "risks_iteration": [],
        "project_risk_score_iterations": [],
    }

    # Stream execution
    config = {"configurable": {"thread_id": "1"}}
    final_state = None

    print(f"\n{YELLOW}{BOLD}🚀 Starting workflow...{RESET}\n")
    for event in graph_plan.stream(state_input, config, stream_mode=["updates"]):
        node_name = next(iter(event[1]))
        print(f"  {GREEN}✅ Completed node:{RESET} {BOLD}{node_name}{RESET}")
        final_state = event[1][node_name]

    print(f"\n{BLUE}─{RESET}" * 60)
    print(f"{GREEN}{BOLD}🎯 Workflow complete!{RESET}")

    # Retrieve the full final state from the graph
    full_state = graph_plan.get_state(config).values

    # Display Gantt chart
    print(f"\n{YELLOW}{BOLD}📊 Generating Gantt chart...{RESET}\n")
    build_gantt_chart(full_state)

# import os

if __name__ == "__main__":
    # print(os.getenv("GROK_API_KEY"))
    main()
