"""
runner/run_system.py
--------------------
Main runtime for ATLAS — handles user interaction, workflow execution,
and Rich-formatted console output.

Public API:
    run_all_system(profile_json, calendar_json, task_json) → (coordinator_output, final_state)
    load_json_and_test()  → (coordinator_output, final_state)
"""

import os
import re
import json
import traceback
from typing import Tuple, Optional, Dict

from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from core.data_manager import DataManager
from graph.workflow import create_agents_graph
from llm.client import get_llm


# ── Main system runner ────────────────────────────────────────────────────────
async def run_all_system(
    profile_json: str,
    calendar_json: str,
    task_json: str,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    End-to-end entry point for the ATLAS academic assistance system.

    Flow:
        1. Initialise Rich console and display welcome banner.
        2. Load student data via DataManager.
        3. Prompt the user for their academic request.
        4. Build the initial AcademicState.
        5. Compile and stream the LangGraph workflow.
        6. Display formatted agent outputs.

    Args:
        profile_json:  Raw JSON string of the student profile.
        calendar_json: Raw JSON string of calendar events.
        task_json:     Raw JSON string of active tasks.

    Returns:
        Tuple of (coordinator_output dict, final_state dict).
        Returns (None, None) on unrecoverable error.
    """
    console = Console()

    try:
        # ── Welcome ────────────────────────────────────────────────────────
        console.print("\n[bold magenta]🎓 ATLAS: Academic Task Learning Agent System[/bold magenta]")
        console.print("[italic blue]Initialising academic support system...[/italic blue]\n")

        # ── Load data ──────────────────────────────────────────────────────
        llm = get_llm()
        dm  = DataManager()
        dm.load_data(profile_json, calendar_json, task_json)

        # ── User input ─────────────────────────────────────────────────────
        console.print("[bold green]Please enter your academic request:[/bold green]")
        user_input = str(input()).strip()
        console.print(f"\n[dim italic]Processing: {user_input}[/dim italic]\n")

        # ── Build initial state ────────────────────────────────────────────
        state = {
            "messages": [HumanMessage(content=user_input)],
            "profile":  dm.get_student_profile("student_123"),
            "calendar": {"events": dm.get_upcoming_events()},
            "tasks":    {"tasks":  dm.get_active_tasks()},
            "results":  {},
        }

        # ── Build workflow graph ───────────────────────────────────────────
        graph = create_agents_graph(llm)
        console.print("[bold cyan]✅ Workflow graph compiled. Processing...[/bold cyan]\n")

        coordinator_output = None
        final_state        = None

        # ── Stream workflow steps ──────────────────────────────────────────
        with console.status("[bold green]Running agents...", spinner="dots"):
            async for step in graph.astream(state):

                # Capture coordinator analysis
                if "coordinator" in step:
                    coordinator_output = step["coordinator"]
                    analysis = coordinator_output.get("results", {}).get(
                        "coordinator_analysis", {}
                    )
                    console.print("\n[bold cyan]📋 Selected Agents:[/bold cyan]")
                    for agent in analysis.get("required_agents", []):
                        console.print(f"  • {agent}")

                # Capture execute step (final outputs)
                if "execute" in step:
                    final_state = step

        # ── Display results ────────────────────────────────────────────────
        _display_results(console, final_state)

        console.print("\n[bold green]✓ Task completed![/bold green]")
        return coordinator_output, final_state

    except Exception as exc:
        console.print(f"\n[bold red]System error:[/bold red] {exc}")
        console.print("[yellow]Stack trace:[/yellow]")
        console.print(traceback.format_exc())
        return None, None

# ── Result display helper ─────────────────────────────────────────────────────
def _display_results(console: Console, final_state: Optional[Dict]) -> None:
    """
    Render each agent's output as a labelled Rich panel.

    Args:
        console:     Rich Console instance.
        final_state: The last streamed workflow step containing agent outputs.
    """
    if not final_state:
        console.print("[yellow]⚠️  No final output to display.[/yellow]")
        return

    agent_outputs = (
        final_state
        .get("execute", {})
        .get("results", {})
        .get("agent_outputs", {})
    )

    for agent_name, output in agent_outputs.items():
        console.print(f"\n[bold cyan]{'─' * 60}[/bold cyan]")
        console.print(f"[bold magenta]🤖 {agent_name.upper()} Output[/bold magenta]")
        console.print(f"[bold cyan]{'─' * 60}[/bold cyan]")

        text = _extract_text(output)
        if text:
            md = Markdown(text)
            panel = Panel(md, border_style="cyan")
            console.print(panel)


def _extract_text(output) -> str:
    """Recursively extract text content from a (possibly nested) output dict."""
    if isinstance(output, str):
        return output.strip()
    if isinstance(output, dict):
        for value in output.values():
            result = _extract_text(value)
            if result:
                return result
    return ""


# ── Local file loader ─────────────────────────────────────────────────────────
async def load_json_and_test() -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Discover and load the three required JSON files from the data/ directory,
    then run the full ATLAS workflow.

    Expected files (matched by regex, case-insensitive):
        profile*.json
        calendar*.json
        task*.json

    Returns:
        Tuple of (coordinator_output, final_state).
    """
    print("Academic Assistant — Test Setup")
    print("-" * 50)

    try:
        # Resolve data directory relative to this file
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        data_dir = os.path.abspath(data_dir)
        current_files = os.listdir(data_dir)

        patterns = {
            "profile":  r"profile.*\.json$",
            "calendar": r"calendar.*\.json$",
            "task":     r"task.*\.json$",
        }

        found_files = {
            file_type: next(
                (f for f in current_files if re.match(pattern, f, re.IGNORECASE)),
                None,
            )
            for file_type, pattern in patterns.items()
        }

        missing = [k for k, v in found_files.items() if v is None]
        if missing:
            print(f"❌ Missing required files: {missing}")
            print(f"   Files found in data/: {current_files}")
            return None, None

        print("\n✅ Files found:")
        for file_type, filename in found_files.items():
            print(f"   {file_type}: {filename}")

        json_contents: Dict[str, str] = {}
        for file_type, filename in found_files.items():
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as fh:
                json_contents[file_type] = fh.read()

        print("\n🚀 Starting ATLAS workflow...\n")
        return await run_all_system(
            json_contents["profile"],
            json_contents["calendar"],
            json_contents["task"],
        )

    except Exception as exc:
        print(f"\n❌ Error: {exc}")
        print(traceback.format_exc())
        return None, None