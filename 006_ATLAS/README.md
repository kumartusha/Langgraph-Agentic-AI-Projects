# 🎓 ATLAS — Academic Task and Learning Agent System

A **production-grade multi-agent AI system** built with **LangGraph + ReACT + Async Python**.

---

## 🏗 Project Structure

```
atlas/
├── main.py                        ← Entry point
├── .env.example                   ← API key template (copy → .env)
├── requirements.txt               ← All dependencies
│
├── config/
│   └── settings.py                ← LLMConfig, API key helpers
│
├── core/
│   ├── state.py                   ← AcademicState TypedDict + dict_reducer
│   └── data_manager.py            ← JSON loader (profile / calendar / tasks)
│
├── llm/
│   └── client.py                  ← NeMoLLaMa (legacy) + get_llm() factory
│
├── agents/
│   ├── base_agent.py              ← ReActAgent base class + Pydantic models
│   ├── coordinator.py             ← Coordinator: decides which agents run
│   ├── profile_analyzer.py        ← Extracts learning patterns from profile
│   ├── planner.py                 ← PlannerAgent (calendar → task → plan)
│   ├── note_writer.py             ← NoteWriterAgent (analyze → generate)
│   └── advisor.py                 ← AdvisorAgent (analyze → guidance)
│
├── executor/
│   └── agent_executor.py          ← Concurrent agent orchestration
│
├── graph/
│   └── workflow.py                ← LangGraph StateGraph wiring
│
├── runner/
│   └── run_system.py              ← Rich UI + workflow streamer
│
└── data/
    ├── profile.json
    ├── calendar_events.json
    └── task.json
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up your API key
```bash
cp .env.example .env
# Edit .env and add your GROK_API_KEY
```

### 3. Run ATLAS
```bash
python main.py
```

---

## 🧠 How It Works

```
User Input
    │
    ▼
Coordinator Agent          ← ReACT: decides which agents to activate
    │
    ▼
Profile Analyzer           ← Extracts learning style & patterns
    │
    ├──► Planner Agent     ← calendar_analyzer → task_analyzer → plan_generator
    ├──► NoteWriter Agent  ← analyze_learning_style → generate_notes
    └──► Advisor Agent     ← analyze_situation → generate_guidance
              │
              ▼
        AgentExecutor      ← runs selected agents concurrently (asyncio.gather)
              │
              ▼
        Rich Console Output
```

### Agent Roles

| Agent | Role |
|---|---|
| **Coordinator** | Routes requests to the right specialist agents |
| **Profile Analyzer** | Reads learning style, study patterns, performance |
| **Planner** | Creates ADHD-aware, energy-optimised study schedules |
| **NoteWriter** | Generates 80/20 study notes tailored to learning style |
| **Advisor** | Provides holistic academic guidance + emergency protocols |

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `GROK_API_KEY` | Groq API key (LLaMA-3.3-70B) — **required** |
| `NEMOTRON_3_NANO_OMNI_30B_A3B_REASONING_KEY` | NVIDIA key — optional/legacy |

---

## 📦 Tech Stack

- **LangGraph** — Multi-agent state machine orchestration
- **LangChain / ChatGroq** — LLM abstraction layer
- **Pydantic** — Agent action/output schema validation
- **asyncio** — Concurrent agent execution
- **Rich** — Beautiful terminal UI
