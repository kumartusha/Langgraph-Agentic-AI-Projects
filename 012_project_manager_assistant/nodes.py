"""
LangGraph workflow node functions for the Project Manager Assistant.

Each function represents a node in the directed graph workflow.
Nodes receive the shared AgentState, invoke the LLM with a
domain-specific prompt, and return state updates.

Node Pipeline:
    task_generation → task_dependency → task_scheduler → task_allocator
    → risk_assessor → (router) → insight_generator → task_scheduler (loop)
"""

import re
import time
from groq import RateLimitError

from config import llm
from state import AgentState
from models import (
    DependencyList,
    RiskList,
    Schedule,
    TaskAllocationList,
    TaskList,
)


def retry_invoke(runnable, prompt, max_retries=3):
    """Invoke a runnable with automatic retry on Groq rate limit errors.
    
    Parses the retry-after time from the error message and waits
    before retrying, up to max_retries attempts.
    """
    for attempt in range(max_retries):
        try:
            return runnable.invoke(prompt)
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            # Extract wait time from error message (e.g., "try again in 35.33s")
            match = re.search(r"try again in (\d+\.?\d*)s", str(e))
            wait_time = float(match.group(1)) + 1 if match else 40
            print(f"    ⏳ Rate limited. Waiting {wait_time:.0f}s before retry {attempt + 2}/{max_retries}...")
            time.sleep(wait_time)


# ──────────────────────────────────────────────
# Node 1: Task Generation
# ──────────────────────────────────────────────

def task_generation_node(state: AgentState):
    """
    Extract actionable tasks from the project description.

    Uses the LLM to analyze the project description and produce a
    structured TaskList. Tasks estimated >5 days are automatically
    broken into smaller sub-tasks.
    """
    description = state["project_description"]
    prompt = f"""
        You are an expert project manager tasked with analyzing the following project description: {description}
        Your objectives are to: 
        1. **Extract Actionable Tasks:**
            - Identify and list all actionable and realistic tasks necessary to complete the project.
            - Provide an estimated number of days required to complete each task.
        2. **Refine Long-Term Tasks:**
            - For any task estimated to take longer than 5 days, break it down into smaller, independent sub-tasks.
        **Requirements:** - Ensure each task is clearly defined and achievable.
            - Maintain logical sequencing of tasks to facilitate smooth project execution.
        Respond with valid JSON only, using exactly the following structure:
        {{
            "tasks": [
                {{
                    "task_name": "Task Name",
                    "task_description": "Detailed description",
                    "estimated_days": 5
                }}
            ]
        }}"""

    structure_llm = llm.with_structured_output(TaskList, method="json_mode")
    tasks: TaskList = retry_invoke(structure_llm, prompt)
    return {"tasks": tasks}


# ──────────────────────────────────────────────
# Node 2: Task Dependency Analysis
# ──────────────────────────────────────────────

def task_dependency_node(state: AgentState):
    """
    Map inter-task dependencies (blocking/dependent relationships).

    For each task, determines which tasks must complete first (blockers)
    and which tasks depend on its completion (downstream).
    """
    tasks = state["tasks"]
    prompt = f"""
        You are a skilled project scheduler responsible for mapping out task dependencies.
        Given the following list of tasks: {tasks}
        Your objectives are to:
            1. **Identify Dependencies:**
                - For each task, determine which other tasks must be completed before it can begin (blocking tasks).
            2. **Map Dependent Tasks:** 
                - For every task, list the NAMES of tasks that depend on its completion.
        IMPORTANT: Use only task NAMES (strings), not full task objects. Keep the output concise.
        Respond with valid JSON only, using exactly the following structure:
        {{
            "dependencies": [
                {{
                    "task_name": "Task Name",
                    "dependent_task_names": ["Dependent Task 1", "Dependent Task 2"]
                }}
            ]
        }}
        """
    structure_llm = llm.with_structured_output(DependencyList, method="json_mode")
    dependencies: DependencyList = retry_invoke(structure_llm, prompt)
    return {"dependencies": dependencies}


# ──────────────────────────────────────────────
# Node 3: Task Scheduling
# ──────────────────────────────────────────────

def task_scheduler_node(state: AgentState):
    """
    Create an optimized project timeline respecting dependencies.

    Assigns start/end days to each task, parallelizing where possible.
    Incorporates insights from previous iterations to improve scheduling.
    """
    dependencies = state["dependencies"]
    tasks = state["tasks"]
    insights = state["insights"]
    prompt = f"""
        You are an experienced project scheduler tasked with creating an optimized project timeline.
        **Given:**
            - **Tasks:** {tasks}
            - **Dependencies:** {dependencies}
            - **Previous Insights:** {insights}
            - **Previous Schedule Iterations (if any):** {state["schedule_iteration"]}
        **Your objectives are to: **
            1. **Develop a Task Schedule:**
                - Assign start and end days to each task, ensuring that all dependencies are respected.
                - Optimize the schedule to minimize the overall project duration.
                - If possible parallelize the tasks to reduce the overall project duration.
                - Try not to increase the project duration compared to previous iterations.
            2. **Incorporate Insights:** 
                - Utilize insights from previous iterations to enhance scheduling efficiency and address any identified issues.
        
        Respond with valid JSON only, using exactly the following structure:
        {{
            "schedule": [
                {{
                    "task": {{
                        "id": "T1",
                        "task_name": "Task Name",
                        "task_description": "Description",
                        "estimated_days": 5
                    }},
                    "start_day": 1,
                    "end_day": 5
                }}
            ]
        }}
        """
    schedule_llm = llm.with_structured_output(Schedule, method="json_mode")
    schedule: Schedule = retry_invoke(schedule_llm, prompt)
    state["schedule"] = schedule
    state["schedule_iteration"].append(schedule)
    return state


# ──────────────────────────────────────────────
# Node 4: Task Allocation
# ──────────────────────────────────────────────

def task_allocation_node(state: AgentState):
    """
    Assign tasks to team members based on skills and availability.

    Ensures no overlapping assignments and balances workload evenly.
    Uses previous iteration insights to improve allocation decisions.
    """
    tasks = state["tasks"]
    schedule = state["schedule"]
    team = state["team"]
    insights = state["insights"]
    prompt = f"""
        You are a proficient project manager responsible for allocating tasks to team members efficiently.
        **Given:** 
            - **Tasks:** {tasks} 
            - **Schedule:** {schedule} 
            - **Team Members:** {team} 
            - **Previous Insights:** {insights} 
            - **Previous Task Allocations (if any):** {state["task_allocations_iteration"]} 
        **Your objectives are to:** 
            1. **Allocate Tasks:** 
                - Assign each task to a team member based on their expertise and current availability. 
                - Ensure that no team member is assigned overlapping tasks during the same time period. 
            2. **Optimize Assignments:** 
                - Utilize insights from previous iterations to improve task allocations. 
                - Balance the workload evenly among team members to enhance productivity and prevent burnout.
                **Constraints:** 
                    - Each team member can handle only one task at a time. 
                    - Assignments should respect the skills and experience of each team member.
        
        Respond with valid JSON only, using exactly the following structure:
        {{
            "task_allocations": [
                {{
                    "task": {{
                        "id": "T1",
                        "task_name": "Task Name",
                        "task_description": "Description",
                        "estimated_days": 5
                    }},
                    "team_member": {{
                        "name": "Alice",
                        "profile": "Developer"
                    }}
                }}
            ]
        }}
        """
    structure_llm = llm.with_structured_output(TaskAllocationList, method="json_mode")
    task_allocations: TaskAllocationList = retry_invoke(structure_llm, prompt)
    state["task_allocations"] = task_allocations
    state["task_allocations_iteration"].append(task_allocations)
    return state


# ──────────────────────────────────────────────
# Node 5: Risk Assessment
# ──────────────────────────────────────────────

def risk_assessment_node(state: AgentState):
    """
    Evaluate risk for each task allocation and schedule pairing.

    Assigns risk scores (0-10) considering task complexity, resource
    availability, seniority, and buffer time. Computes the overall
    project risk score as the sum of individual scores.
    """
    schedule = state["schedule"]
    task_allocations = state["task_allocations"]
    prompt = f"""
        You are a seasoned project risk analyst tasked with evaluating the risks associated with the current project plan.
        **Given:**
            - **Task Allocations:** {task_allocations}
            - **Schedule:** {schedule}
            - **Previous Risk Assessments (if any):** {state['risks_iteration']}
        **Your objectives are to:**
            1. **Assess Risks:**
                - Analyze each allocated task and its scheduled timeline to identify potential risks.
                - Consider factors such as task complexity, resource availability, and dependency constraints.
            2. **Assign Risk Scores:**
            - Assign a risk score to each task on a scale from 0 (no risk) to 10 (high risk).
            - If a task assignment remains unchanged from a previous iteration (same team member and task), retain the existing risk score to ensure consistency.
            - If the team member has more time between tasks - assign lower risk score for the tasks
            - If the task is assigned to a more senior person - assign lower risk score for the tasks
            3. **Calculate Overall Project Risk:**
            - Sum the individual task risk scores to determine the overall project risk score.
        
        Respond with valid JSON only, using exactly the following structure:
        {{
            "risks": [
                {{
                    "task": {{
                        "id": "T1",
                        "task_name": "Task Name",
                        "task_description": "Description",
                        "estimated_days": 5
                    }},
                    "score": "5"
                }}
            ]
        }}
        """
    structure_llm = llm.with_structured_output(RiskList, method="json_mode")
    risks: RiskList = retry_invoke(structure_llm, prompt)

    # Compute aggregate project risk score
    project_task_risk_scores = [int(risk.score) for risk in risks.risks]
    project_risk_score = sum(project_task_risk_scores)

    state["risks"] = risks
    state["project_risk_score"] = project_risk_score
    state["iteration_number"] += 1
    state["project_risk_score_iterations"].append(project_risk_score)
    state["risks_iteration"].append(risks)
    return state


# ──────────────────────────────────────────────
# Node 6: Insight Generation
# ──────────────────────────────────────────────

def insight_generation_node(state: AgentState):
    """
    Generate actionable improvement insights for the next iteration.

    Analyzes current allocations, schedule, and risks to recommend
    bottleneck resolution, resource rebalancing, and risk mitigation.
    """
    schedule = state["schedule"]
    task_allocations = state["task_allocations"]
    risks = state["risks"]
    prompt = f"""
        You are an expert project manager responsible for generating actionable insights to enhance the project plan.
        **Given:**
            - **Task Allocations:** {task_allocations}
            - **Schedule:** {schedule}
            - **Risk Analysis:** {risks}
        **Your objectives are to:**
            1. **Generate Critical Insights:**
            - Analyze the current task allocations, schedule, and risk assessments to identify areas for improvement.
            - Highlight any potential bottlenecks, resource conflicts, or high-risk tasks that may jeopardize project success.
            2. **Recommend Enhancements:**
            - Suggest adjustments to task assignments or scheduling to mitigate identified risks.
            - Propose strategies to optimize resource utilization and streamline workflow.
                **Requirements:**
                - Ensure that all recommendations aim to reduce the overall project risk score.
                - Provide clear and actionable suggestions that can be implemented in subsequent iterations.
        """
    insights = retry_invoke(llm, prompt).content
    return {"insights": insights}
