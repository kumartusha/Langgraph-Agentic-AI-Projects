"""
executor/agent_executor.py
--------------------------
AgentExecutor — concurrent orchestration layer for ATLAS specialist agents.

Reads the coordination plan produced by the CoordinatorAgent and executes
the required specialist agents (PLANNER, NOTEWRITER, ADVISOR) in parallel
groups using asyncio.gather().

Fallback strategy:
    - If a group produces no results, falls back to PLANNER.
    - If everything fails, returns a safe emergency response.

Public API:
    AgentExecutor(llm)
    AgentExecutor.execute(state) → Dict   ← LangGraph node callable
"""

import asyncio
from typing import Dict

from core.state import AcademicState
from agents.planner import PlannerAgent
from agents.note_writer import NoteWriterAgent
from agents.advisor import AdvisorAgent


class AgentExecutor:
    """
    Orchestrates concurrent execution of multiple specialist agents.

    Agents are instantiated once at construction and reused across calls.

    Registered agents:
        PLANNER    → PlannerAgent
        NOTEWRITER → NoteWriterAgent
        ADVISOR    → AdvisorAgent
    """

    def __init__(self, llm):
        """
        Args:
            llm: Shared language model instance passed to all agents.
        """
        self.llm = llm
        self.agents = {
            "PLANNER":    PlannerAgent(llm),
            "NOTEWRITER": NoteWriterAgent(llm),
            "ADVISOR":    AdvisorAgent(llm),
        }

    async def execute(self, state: AcademicState) -> Dict:
        """
        Run required agents concurrently based on the coordinator's analysis.

        Execution strategy:
            1. Read coordinator_analysis to get required_agents + concurrent_groups.
            2. For each concurrent group, gather tasks from available agents.
            3. Run each group in parallel; collect successful results.
            4. If no results → fall back to PLANNER.
            5. On total failure → return emergency fallback response.

        Args:
            state: Current AcademicState with coordinator_analysis populated.

        Returns:
            Dict to merge into state["results"]["agent_outputs"].
        """
        try:
            analysis         = state["results"].get("coordinator_analysis", {})
            required_agents  = analysis.get("required_agents", ["PLANNER"])
            concurrent_groups = analysis.get("concurrent_groups", [])

            results: Dict = {}

            # ── Execute each concurrent group ──────────────────────────────
            for group in concurrent_groups:
                tasks = []
                valid_agents = []

                for agent_name in group:
                    if agent_name in required_agents and agent_name in self.agents:
                        tasks.append(self.agents[agent_name](state))
                        valid_agents.append(agent_name)

                if not tasks:
                    continue

                # Run agents in this group concurrently
                group_results = await asyncio.gather(*tasks, return_exceptions=True)

                for agent_name, result in zip(valid_agents, group_results):
                    if isinstance(result, Exception):
                        print(f"⚠️  {agent_name} failed: {result}")
                    else:
                        results[agent_name.lower()] = result

            # ── Fallback: run PLANNER if nothing succeeded ─────────────────
            if not results and "PLANNER" in self.agents:
                print("ℹ️  No agent results — falling back to PLANNER.")
                planner_result = await self.agents["PLANNER"](state)
                results["planner"] = planner_result

            print(f"✅ Agent outputs: {list(results.keys())}")

            return {
                "results": {
                    "agent_outputs": results
                }
            }

        except Exception as exc:
            print(f"❌ AgentExecutor fatal error: {exc}")
            return {
                "results": {
                    "agent_outputs": {
                        "planner": {
                            "plan": "Emergency fallback: please try again or contact support."
                        }
                    }
                }
            }
