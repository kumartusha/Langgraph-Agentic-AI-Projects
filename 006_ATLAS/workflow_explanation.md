# 🎓 ATLAS Workflow — Complete Theoretical Walkthrough

> **Example Request used throughout:** *"I have a Calculus III exam tomorrow and I can't focus. Help!"*

---

## 📦 The Big Picture

```
User types a message
        ↓
  [ LangGraph StateGraph ] — the "traffic controller"
        ↓
  Passes a shared "AcademicState" box through every step
  (like a baton in a relay race — everyone reads & writes to it)
```

Think of **AcademicState** as a shared whiteboard that every agent reads from and writes to:

```python
state = {
    "messages":  [HumanMessage("I have Calculus III exam tomorrow...")],
    "profile":   { name: "Alex", learning_style: "visual", ... },
    "calendar":  { events: [...] },
    "tasks":     { tasks: [...] },
    "results":   {}   # ← starts empty, fills up as agents run
}
```

---

## 🗺️ Complete Flow at a Glance

```
User: "Calculus III exam tomorrow, can't focus!"
           │
           ▼
    [main.py] → load JSON files → build state
           │
           ▼
    [coordinator] → "need PLANNER + NOTEWRITER" → writes to state.results
           │
           ▼
    [profile_analyzer] → "Alex = visual learner, 45-min sprints" → writes to state.results
           │
     ┌─────┴──────┐  ← PARALLEL (asyncio.gather)
     ▼            ▼
 [calendar_    [notewriter_
  analyzer]     analyze]
     ↓            ↓
 [task_        [notewriter_
  analyzer]     generate]
     ↓            ↓
 [plan_        "Color-coded
  generator]   Calculus notes"
 "9AM sprints"
     └─────┬──────┘
           ▼
      [execute] → combines both outputs safely
           │
           ▼
      should_end? → YES (all required agents done)
           │
           ▼
     [Rich console] → displays plan + notes in panels
```

---

## Step-by-Step Flow

---

### 🟢 STEP 0 — `main.py` → Entry Point

**File:** `main.py`

```bash
python main.py
```

- Calls `configure_api_keys()` → checks `.env` for `GROK_API_KEY`
- Calls `load_json_and_test()` from `runner/run_system.py`
- That function finds `profile.json`, `calendar_events.json`, `task.json` from `data/`
- Calls `run_all_system()` which builds the state and fires the graph

> **Why?** You need one clean entry point. Without this, you'd manually wire everything every time.

---

### 🟡 STEP 1 — `DataManager` → Loads Real Data

**File:** `core/data_manager.py`

```python
dm = DataManager()
dm.load_data(profile_json, calendar_json, task_json)

profile  = dm.get_student_profile("student_123")
events   = dm.get_upcoming_events()   # next 7 days only
tasks    = dm.get_active_tasks()      # status="needsAction" + future due date
```

**What it does:**
- Parses the 3 JSON files
- Filters calendar → only future events (not past ones)
- Filters tasks → only incomplete + not overdue
- Builds the initial `state` dict

> **Why?** Agents shouldn't work with raw messy JSON. DataManager gives clean, filtered data.

---

### 🔵 STEP 2 — `coordinator_agent` → The Brain Decides

**File:** `agents/coordinator.py`

This is the **first LangGraph node** to run (`START → coordinator`).

**What it does:**

**1. Calls `analyze_context(state)`** → scans the profile + request and builds a summary:

```python
context = {
    "student": { "major": "Computer Science", "learning_style": "visual" },
    "course":  { "name": "Calculus III", "performance": "B" },  # matched from message
    "upcoming_events": 3,
    "active_tasks":    2
}
```

**2. Sends this to the LLM (ChatGroq)** with the `COORDINATOR_PROMPT`:

```
"User wants help with Calculus III exam tomorrow.
Student is a visual learner with 2 pending tasks.
Which agents do you need: PLANNER, NOTEWRITER, ADVISOR?"
```

**3. LLM responds with ReACT pattern:**

```
Thought:  Student has an emergency exam. Needs quick notes + a study plan.
Action:   Deploy PLANNER + NOTEWRITER in parallel. ADVISOR optional.
Decision: required_agents = [PLANNER, NOTEWRITER]
```

**4. `parse_coordinator_response()`** reads that text and extracts:

```python
{
    "required_agents":   ["PLANNER", "NOTEWRITER"],
    "concurrent_groups": [["PLANNER", "NOTEWRITER"]],
    "priority":          {"PLANNER": 1, "NOTEWRITER": 2}
}
```

**5. Writes** this into `state["results"]["coordinator_analysis"]`

> **Why?** Not every request needs all 3 agents. A simple scheduling question doesn't need notes. This step avoids wasting LLM calls.

---

### 🔵 STEP 3 — `profile_analyzer` → Reads the Student

**File:** `agents/profile_analyzer.py`

Runs right after coordinator (`coordinator → profile_analyzer`).

**What it does:**

Sends the full `profile.json` to the LLM with the `PROFILE_ANALYZER_PROMPT`:

```
"Analyse Alex's profile:
 - Visual learner, prefers mornings
 - 45-min sessions, 10-min breaks
 - Optimal environment: library, low noise
 - Focus score: 0.75 (target 0.85)"
```

LLM responds:

```
Observation: Alex is a visual learner who works best in 45-min sprints.
             Morning sessions are optimal. Has ADHD-like focus challenges.
Recommendations: Use diagrams, color-coding, mind maps for Calculus.
```

Writes this into `state["results"]["profile_analysis"]`

> **Why?** Every downstream agent (Planner, NoteWriter, Advisor) needs to know HOW this specific student learns before giving advice. This runs once and everyone benefits.

---

### 🟣 STEP 4 — Parallel Routing → Fork the Road

**File:** `graph/workflow.py` — `route_to_parallel_agents()`

```python
def route_to_parallel_agents(state):
    required = ["PLANNER", "NOTEWRITER"]   # from coordinator
    return ["calendar_analyzer", "notewriter_analyze"]  # both start simultaneously
```

LangGraph **forks** here — two parallel paths start at the same time:

```
profile_analyzer
    ├──────────────────────────────┐
    ↓                              ↓
calendar_analyzer          notewriter_analyze
    ↓                              ↓
task_analyzer              notewriter_generate
    ↓                              ↓
plan_generator              (writes to state["results"]["generated_notes"])
    ↓
(writes to state["results"]["final_plan"])
    └──────────────────────────────┘
                    ↓
                 execute
```

> **Why?** The Planner and NoteWriter don't depend on each other. Running them simultaneously saves time — otherwise they'd run sequentially and double the wait.

---

### 🟣 STEP 5A — `PlannerAgent` → 3-Step Sub-Workflow

**File:** `agents/planner.py`

This agent has its **OWN internal mini-graph:** `calendar_analyzer → task_analyzer → plan_generator`

#### Sub-step 5A-1: `calendar_analyzer`

- Reads `state["calendar"]["events"]`
- Filters to next 7 days
- Sends to LLM: *"What time blocks are free? What conflicts exist?"*
- LLM finds: *"Tomorrow 9AM–1PM is free. Football match at 6PM."*
- Writes → `state["results"]["calendar_analysis"]`

#### Sub-step 5A-2: `task_analyzer`

- Reads `state["tasks"]["tasks"]`
- Sends to LLM: *"Prioritise these tasks by urgency/complexity"*
- LLM output: *"Calculus III exam is highest priority. Graph Theory project due in 3 days."*
- Writes → `state["results"]["task_analysis"]`

#### Sub-step 5A-3: `plan_generator`

Reads **ALL THREE** previous analyses:
- `profile_analysis` → visual learner, 45-min sprints
- `calendar_analysis` → free 9AM–1PM tomorrow
- `task_analysis` → Calculus is top priority

Uses few-shot examples as guidance, sends everything to LLM:

```
"Create an ADHD-friendly study plan for tomorrow's Calculus III exam.
 Student is a visual learner. Free time: 9AM–1PM. Football at 6PM."
```

LLM generates:

```
EMERGENCY STUDY PLAN:
  9:00–9:45AM  → Integration techniques (mind maps + diagrams)
  9:45–9:55AM  → Break (walk around)
  9:55–10:40AM → Vector calculus practice problems
  10:40–11:00AM → Long break (snack, move)
  ...
  EMERGENCY PROTOCOLS:
    - Losing focus → switch topics immediately
    - Overwhelmed → 5-min cold water break
```

Writes → `state["results"]["final_plan"]`

---

### 🟣 STEP 5B — `NoteWriterAgent` → 2-Step Sub-Workflow

**File:** `agents/note_writer.py`

Runs **at the same time as PlannerAgent** (parallel).

#### Sub-step 5B-1: `analyze_learning_style`

- Reads profile's learning style: *"visual, mind mapping, Khan Academy"*
- Asks LLM: *"How should I structure Calculus III notes for a visual learner under time pressure?"*
- LLM: *"Use color-coded quick reference cards. Focus on the 20% of formulas used in 80% of exam problems."*
- Writes → `state["results"]["learning_analysis"]`

#### Sub-step 5B-2: `generate_notes`

Reads `learning_analysis` + request + few-shot example templates.
Asks LLM to generate actual study notes:

```
CALCULUS III EMERGENCY NOTES:

🟦 CORE FORMULAS (Must Know):
  • ∭f dV in cylindrical = ∫∫∫ f(r,θ,z) r dz dr dθ
  • Curl F = ∇ × F
  • Divergence Theorem: ∯ F·dS = ∭ (∇·F) dV

🟨 VISUAL MEMORY TRICKS:
  • Think of divergence as "water flowing out of a sponge"
  • Curl = rotation of a paddle wheel in the field

🟩 EXAM PATTERN (Last 5 years):
  • Q1 always: Find critical points (set ∇f = 0)
  • Q2 always: Calculate flux through a surface
```

Writes → `state["results"]["generated_notes"]`

---

### 🔴 STEP 6 — `AgentExecutor` → Collect & Combine

**File:** `executor/agent_executor.py`

After both parallel paths finish, they both feed into **execute**.

**What it does:**

```python
# Reads coordinator's plan
required_agents   = ["PLANNER", "NOTEWRITER"]
concurrent_groups = [["PLANNER", "NOTEWRITER"]]

# Runs them with asyncio.gather() — truly concurrent
results = await asyncio.gather(
    planner_agent(state),
    notewriter_agent(state),
    return_exceptions=True   # one failing doesn't kill the other
)

# Combines into one dict
agent_outputs = {
    "planner":    { "plan": "9AM sprint schedule..." },
    "notewriter": { "notes": "Calculus III essentials..." }
}
```

- If everything fails → fallback to PLANNER only
- Writes → `state["results"]["agent_outputs"]`

> **Why?** You need one place that safely gathers everything. If NoteWriter crashes, the Planner's output still gets returned.

---

### 🟢 STEP 7 — `should_end` → Loop or Finish?

**File:** `graph/workflow.py`

```python
def should_end(state):
    executed = {"planner", "notewriter"}   # what we got back
    required = {"planner", "notewriter"}   # what coordinator asked for

    if required.issubset(executed):
        return END   # ✅ all done
    else:
        return "coordinator"   # 🔁 loop back, something is missing
```

In our case: `{"planner","notewriter"} ⊆ {"planner","notewriter"}` → **END** ✅

> **Why?** Fault tolerance. If an agent silently fails and returns nothing, the graph loops back to the coordinator to retry instead of silently producing incomplete output.

---

### 🖥️ STEP 8 — `run_system.py` → Display to User

**File:** `runner/run_system.py`

```python
# Streams each step in real time
async for step in graph.astream(state):
    if "coordinator" in step:
        print("Selected agents: PLANNER, NOTEWRITER")
    if "execute" in step:
        # final outputs available — render with Rich

# Displays using Rich panels:
╭─────────────────────╮
│  🤖 PLANNER Output  │
│  9AM–9:45 Calculus  │
│  ...                │
╰─────────────────────╯
╭────────────────────────╮
│  🤖 NOTEWRITER Output  │
│  ∭f dV = ...          │
╰────────────────────────╯
```

> **Why streaming?** `graph.astream()` lets you show results as they come in — the user sees progress instead of staring at a blank screen for 30 seconds.

---

## 📊 State Evolution — What the Whiteboard Looks Like at Each Step

| After Step | `state["results"]` contains |
|---|---|
| Start | `{}` (empty) |
| After Coordinator | `coordinator_analysis: { required_agents, concurrent_groups, reasoning }` |
| After Profile Analyzer | `+ profile_analysis: { analysis }` |
| After Calendar Analyzer | `+ calendar_analysis: { analysis }` |
| After Task Analyzer | `+ task_analysis: { analysis }` |
| After Plan Generator | `+ final_plan: { plan }` |
| After NoteWriter Analyze | `+ learning_analysis: { analysis }` |
| After NoteWriter Generate | `+ generated_notes: { notes }` |
| After Execute | `+ agent_outputs: { planner: {...}, notewriter: {...} }` |

---

## 🧠 Why Each Component Exists

| Component | Real-World Analogy |
|---|---|
| `AcademicState` | Shared Google Doc everyone reads & edits |
| `DataManager` | Librarian who finds and pre-filters your books |
| `Coordinator` | Manager who reads your email and assigns tasks to the right team members |
| `ProfileAnalyzer` | HR reading your profile before assigning the right team |
| `PlannerAgent` | Project manager who checks your calendar and builds a timeline |
| `NoteWriterAgent` | Tutor who writes flashcards tailored to how you learn |
| `AdvisorAgent` | Counsellor who gives holistic life + study advice |
| `AgentExecutor` | Team lead who collects everyone's work and handles failures |
| `LangGraph StateGraph` | The entire office building — defines who sits where and who talks to whom |

---

## 🔑 Key Design Decisions Explained

### 1. Why LangGraph instead of plain Python?
Plain Python can't do conditional routing, parallel execution, and loop-back recovery cleanly. LangGraph handles all of this as a compiled graph with clear visual structure.

### 2. Why `dict_reducer` in the state?
LangGraph merges state updates from all nodes. Without `dict_reducer`, a node writing `{"results": {"plan": "..."}}` would **overwrite** the entire results dict, deleting everything written by previous nodes. The reducer does a deep merge instead.

### 3. Why does each agent have its own sub-graph?
Each agent's internal steps (e.g., calendar → task → plan) are independent workflows. Packaging them as sub-graphs means they can be tested and run in isolation, and re-used in future projects.

### 4. Why `asyncio.gather` in AgentExecutor?
Without it, PLANNER runs first (30 seconds), then NOTEWRITER (30 seconds) = 60 seconds total. With `gather`, they run simultaneously = ~30 seconds. Same output, half the time.

### 5. Why the loop-back `should_end` check?
Resilience. If an agent crashes silently, the coordinator gets a second chance to re-route. The system degrades gracefully instead of returning incomplete results.
