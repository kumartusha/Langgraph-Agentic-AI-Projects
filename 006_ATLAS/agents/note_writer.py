"""
agents/note_writer.py
---------------------
NoteWriter Agent — generates personalised, learning-style-adapted study notes.

Internal subgraph:
    notewriter_analyze → notewriter_generate

The 80/20 principle guides content selection: focus on the 20% of concepts
that give 80% of the understanding.

Public API:
    NoteWriterAgent(llm)
    NoteWriterAgent.__call__(state) → Dict
    NoteWriterAgent.analyze_learning_style(state) → Dict
    NoteWriterAgent.generate_notes(state) → Dict
"""

import json
from typing import Dict

from langgraph.graph import StateGraph, END, START

from agents.base_agent import ReActAgent
from core.state import AcademicState


# ── Few-shot examples ─────────────────────────────────────────────────────────
_FEW_SHOT_EXAMPLES = [
    {
        "input":    "Need to cram Calculus III for tomorrow",
        "template": "Quick Review",
        "notes": """\
CALCULUS III ESSENTIALS:

  1. CORE CONCEPTS (80/20 Rule):
     • Multiple Integrals → volume / area
     • Vector Calculus    → flow / force / rotation
     KEY FORMULAS:
       - Triple integrals in cylindrical / spherical coords
       - Curl, divergence, gradient relationships

  2. COMMON EXAM PATTERNS:
     • Find critical points
     • Calculate flux / work
     • Optimise with constraints

  3. QUICKSTART GUIDE:
     • Always draw 3-D diagrams
     • Check units match
     • Use symmetry to simplify

  4. EMERGENCY TIPS:
     • If stuck → try converting coordinates
     • Check boundary conditions
     • Look for special patterns""",
    }
]


# ── NoteWriterAgent ───────────────────────────────────────────────────────────
class NoteWriterAgent(ReActAgent):
    """
    Specialist agent for creating ADHD-friendly study materials.

    Internal workflow:
        1. analyze_learning_style — determine optimal note structure.
        2. generate_notes         — produce concise, high-impact notes.
    """

    def __init__(self, llm):
        super().__init__(llm)
        self.llm = llm
        self.few_shot_examples = _FEW_SHOT_EXAMPLES
        self.workflow = self._build_subgraph()

    # ── Subgraph ──────────────────────────────────────────────────────────────
    def _build_subgraph(self) -> StateGraph:
        """Wire the NoteWriter sub-graph and compile it."""
        sg = StateGraph(AcademicState)
        sg.add_node("notewriter_analyze",  self.analyze_learning_style)
        sg.add_node("notewriter_generate", self.generate_notes)

        sg.add_edge(START, "notewriter_analyze")
        sg.add_edge("notewriter_analyze",  "notewriter_generate")
        sg.add_edge("notewriter_generate", END)
        return sg.compile()

    # ── Node 1: Analyse Learning Style ────────────────────────────────────────
    async def analyze_learning_style(self, state: AcademicState) -> Dict:
        """
        Determine the optimal note structure for this student.

        Considers:
            - Primary and secondary learning styles.
            - Student's content request and time constraints.

        Returns:
            state update with results["learning_analysis"].
        """
        profile        = state["profile"]
        learning_style = profile["learning_preferences"]["learning_style"]

        prompt = """\
Analyse content requirements and determine the optimal note structure:

STUDENT PROFILE:
  Learning Style: {style}
  Request:        {request}

FORMAT:
  1. Key Topics (80/20 principle)
  2. Learning Style Adaptations
  3. Time Management Strategy
  4. Quick Reference Format

FOCUS ON:
  - Essential concepts that give maximum understanding
  - Visual and interactive elements
  - Time-optimised study methods
""".format(
            style=json.dumps(learning_style, indent=2),
            request=state["messages"][-1].content,
        )

        response = (await self.llm.ainvoke(prompt)).content

        return {
            "results": {
                "learning_analysis": {"analysis": response}
            }
        }

    # ── Node 2: Generate Notes ────────────────────────────────────────────────
    async def generate_notes(self, state: AcademicState) -> Dict:
        """
        Produce structured, personalised study notes.

        Based on:
            - Learning style analysis from the previous node.
            - Few-shot examples for format guidance.

        Returns:
            state update with results["generated_notes"].
        """
        analysis       = state["results"].get("learning_analysis", "")
        learning_style = state["profile"]["learning_preferences"]["learning_style"]

        prompt = """\
Create concise, high-impact study materials based on the analysis:

ANALYSIS:       {analysis}
LEARNING STYLE: {style}
REQUEST:        {request}

EXAMPLES:
{examples}

FORMAT:
  **THREE-WEEK INTENSIVE STUDY PLANNER**

  Generate structured notes with:
    1. Weekly breakdown
    2. Daily focus areas
    3. Core concepts
    4. Emergency tips
""".format(
            analysis=analysis,
            style=json.dumps(learning_style, indent=2),
            request=state["messages"][-1].content,
            examples=json.dumps(self.few_shot_examples, indent=2),
        )

        response = (await self.llm.ainvoke(prompt)).content

        return {
            "results": {
                "generated_notes": {"notes": response}
            }
        }

    # ── Entry Point ───────────────────────────────────────────────────────────
    async def __call__(self, state: AcademicState) -> Dict:
        """Run the full NoteWriter subgraph and return generated notes."""
        try:
            final_state = await self.workflow.ainvoke(state)
            return {"notes": final_state["results"].get("generated_notes")}
        except Exception as exc:
            print(f"❌ NoteWriterAgent error: {exc}")
            return {"notes": "Error generating notes. Please try again."}
