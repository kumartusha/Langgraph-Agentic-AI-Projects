"""
agents/profile_analyzer.py
--------------------------
Profile Analyzer Agent — deep-reads the student profile and extracts
learning patterns that inform downstream planning agents.

Public API:
    profile_analyzer(state) → LangGraph node callable.
"""

import json
from typing import Dict

from core.state import AcademicState


# ── Prompt ────────────────────────────────────────────────────────────────────
PROFILE_ANALYZER_PROMPT = """\
You are a Profile Analysis Agent using the ReACT framework to analyse student profiles.

OBJECTIVE:
  Analyse the student profile and extract key learning patterns that will impact
  their academic success.

REACT PATTERN:
  Thought:     Analyse what aspects of the profile need investigation
  Action:      Extract specific information from relevant profile sections
  Observation: Note key patterns and implications
  Response:    Provide structured analysis

PROFILE DATA:
  {profile}

ANALYSIS FRAMEWORK:
  1. Learning Characteristics
     • Primary learning style
     • Information processing patterns
     • Attention span characteristics

  2. Environmental Factors
     • Optimal study environment
     • Distraction triggers
     • Productive time periods

  3. Executive Function
     • Task management patterns
     • Focus duration limits
     • Break requirements

  4. Energy Management
     • Peak energy periods
     • Recovery patterns
     • Fatigue signals

INSTRUCTIONS:
  1. Use the ReACT pattern for each analysis area.
  2. Provide specific, actionable observations.
  3. Note both strengths and challenges.
  4. Identify patterns that affect study planning.

FORMAT YOUR RESPONSE AS:
  Thought:          [Initial analysis of profile components]
  Action:           [Specific areas being examined]
  Observation:      [Patterns and insights discovered]
  Analysis Summary: [Structured breakdown of key findings]
  Recommendations:  [Specific adaptations needed]
"""


# ── LangGraph Node ────────────────────────────────────────────────────────────
async def profile_analyzer(state: AcademicState) -> Dict:
    """
    Analyse the student profile and enrich state with learning insights.

    This node runs immediately after the coordinator and provides
    personalised learning context for all downstream agents.

    Args:
        state: AcademicState containing student profile data.

    Returns:
        Dict to merge into state["results"]["profile_analysis"].
    """
    from llm.client import get_llm
    llm = get_llm()

    profile = state["profile"]

    prompt = PROFILE_ANALYZER_PROMPT.format(profile=json.dumps(profile, indent=2))

    # Use ainvoke() for non-blocking async LLM call
    response = (await llm.ainvoke(prompt)).content

    return {
        "results": {
            "profile_analysis": {
                "analysis": response
            }
        }
    }
