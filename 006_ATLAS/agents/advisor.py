"""
agents/advisor.py
-----------------
Advisor Agent — provides personalised academic guidance and stress management.

Internal subgraph:
    advisor_analyze → advisor_generate

The advisor synthesises the student's full context (profile + request) into
actionable, compassionate advice with clear emergency protocols.

Public API:
    AdvisorAgent(llm)
    AdvisorAgent.__call__(state) → Dict
    AdvisorAgent.analyze_situation(state) → Dict
    AdvisorAgent.generate_guidance(state) → Dict
"""

import json
from typing import Dict

from langgraph.graph import StateGraph, END, START

from agents.base_agent import ReActAgent
from core.state import AcademicState


# ── Few-shot examples ─────────────────────────────────────────────────────────
_FEW_SHOT_EXAMPLES = [
    {
        "request": "Managing multiple deadlines with limited time",
        "profile": {
            "learning_style":    "visual",
            "workload":          "heavy",
            "time_constraints":  ["2 hackathons", "project", "exam"],
        },
        "advice": """\
PRIORITY-BASED SCHEDULE:

  1. IMMEDIATE ACTIONS
     • Create a visual timeline of all deadlines
     • Break each task into 45-min chunks
     • Schedule high-focus work in the mornings

  2. WORKLOAD MANAGEMENT
     • Hackathons: Form team early, set clear roles
     • Project: Daily 2-hour focused sessions
     • Exam: Interleaved practice with breaks

  3. ENERGY OPTIMISATION
     • Use Pomodoro (25/5) for intensive tasks
     • Physical activity between study blocks
     • Regular progress tracking

  4. EMERGENCY PROTOCOLS
     • If overwhelmed → take a 10-min reset break
     • If stuck       → switch tasks or environments
     • If tired       → quick power nap, then review""",
    }
]


# ── AdvisorAgent ──────────────────────────────────────────────────────────────
class AdvisorAgent(ReActAgent):
    """
    Specialist agent for personalised academic guidance and wellbeing support.

    Internal workflow:
        1. analyze_situation  — understand the student's challenges.
        2. generate_guidance  — produce structured, actionable advice.
    """

    def __init__(self, llm):
        super().__init__(llm)
        self.llm = llm
        self.few_shot_examples = _FEW_SHOT_EXAMPLES
        self.workflow = self._build_subgraph()

    # ── Subgraph ──────────────────────────────────────────────────────────────
    def _build_subgraph(self) -> StateGraph:
        """Wire the Advisor sub-graph and compile it."""
        sg = StateGraph(AcademicState)
        sg.add_node("advisor_analyze",  self.analyze_situation)
        sg.add_node("advisor_generate", self.generate_guidance)

        sg.add_edge(START, "advisor_analyze")
        sg.add_edge("advisor_analyze",  "advisor_generate")
        sg.add_edge("advisor_generate", END)
        return sg.compile()

    # ── Node 1: Analyse Situation ─────────────────────────────────────────────
    async def analyze_situation(self, state: AcademicState) -> Dict:
        """
        Deeply understand the student's current challenges and constraints.

        Evaluates:
            - Learning style compatibility with the current request.
            - Time management and stress management needs.

        Returns:
            state update with results["situation_analysis"].
        """
        profile       = state["profile"]
        learning_prefs = profile.get("learning_preferences", {})

        prompt = """\
Analyse the student situation and determine the guidance approach:

CONTEXT:
  Profile:             {profile}
  Learning Preferences:{prefs}
  Request:             {request}

ANALYSE:
  1. Current challenges
  2. Learning style compatibility
  3. Time management needs
  4. Stress management requirements
""".format(
            profile=json.dumps(profile, indent=2),
            prefs=json.dumps(learning_prefs, indent=2),
            request=state["messages"][-1].content,
        )

        response = (await self.llm.ainvoke(prompt)).content

        return {
            "results": {
                "situation_analysis": {"analysis": response}
            }
        }

    # ── Node 2: Generate Guidance ─────────────────────────────────────────────
    async def generate_guidance(self, state: AcademicState) -> Dict:
        """
        Produce structured, actionable academic guidance.

        Covers:
            - Immediate action steps.
            - Schedule optimisation.
            - Energy and resource management.
            - Support strategies and contingency planning.

        Returns:
            state update with results["guidance"].
        """
        analysis = state["results"].get("situation_analysis", "")

        prompt = """\
Generate personalised academic guidance based on the analysis:

ANALYSIS: {analysis}
EXAMPLES: {examples}

FORMAT:
  1. Immediate Action Steps
  2. Schedule Optimisation
  3. Energy Management
  4. Support Strategies
  5. Emergency Protocols
""".format(
            analysis=analysis,
            examples=json.dumps(self.few_shot_examples, indent=2),
        )

        response = (await self.llm.ainvoke(prompt)).content

        return {
            "results": {
                "guidance": {"advice": response}
            }
        }

    # ── Entry Point ───────────────────────────────────────────────────────────
    async def __call__(self, state: AcademicState) -> Dict:
        """Run the full Advisor subgraph and return guidance with metadata."""
        try:
            final_state = await self.workflow.ainvoke(state)
            return {
                "advisor_output": {
                    "guidance": final_state["results"].get("guidance"),
                    "metadata": {
                        "course_specific":         True,
                        "considers_learning_style": True,
                    },
                }
            }
        except Exception as exc:
            print(f"❌ AdvisorAgent error: {exc}")
            return {
                "advisor_output": {
                    "guidance": "Error generating guidance. Please try again."
                }
            }
