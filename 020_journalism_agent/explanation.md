# Journalism-Focused AI Assistant API (020)

## 📖 Project Overview

This project transforms the experimental "Journalism-Focused AI Assistant" Jupyter Notebook into a **Clean Monolithic CLI application** utilizing LangGraph and the Groq LLM API. The system provides tools for journalists, including summarization, fact-checking (via DuckDuckGo searches), tone analysis, quote extraction, and grammar & bias review.

By moving this to a modular layered architecture, it allows journalists and developers to easily integrate fact-checking and automated review pipelines into CMS platforms, editorial workflows, or larger automated media tools.

---

## 🎯 Problem, Solution, and Impact

### The Problem
Journalism faces massive challenges with the proliferation of online misinformation, subtle biases, and information overload. Reporters often have to manually sift through enormous articles, verify obscure claims, and meticulously check their own text for unintentional bias—a time-consuming process under tight deadlines.

### The Solution
A LangGraph-powered orchestrator that dynamically analyzes text. A user provides a prompt (e.g. "Can you provide a full report on this article?"). The system uses an **Intent Classifier** (Categorize node) to determine which specific sub-agents need to run:
1. **Summarization Agent**: Condenses long-form text.
2. **Fact-Checking Agent**: Verifies claims against DuckDuckGo web searches.
3. **Tone Analysis Agent**: Detects sentiment (positive, neutral, critical).
4. **Reviewer Agent**: Extracts direct quotes and reviews text for grammar and bias.

### The Impact
Significantly reduces the time journalists spend fact-checking and editing. By running articles through this system, editorial teams can ensure their claims are backed by external evidence, their tone remains objective, and their quotes are properly extracted—all through an automated AI pipeline.

---

## 🏗️ Technical Architecture

The project adheres to a Clean Architecture model:

1. **`main.py` (Orchestrator)**: CLI entry point handling PDF text extraction and user input.
2. **`src/agents/` (Domain Logic)**:
   - `router.py`: Categorizes the user's query into actionable system commands.
   - `summarizer.py`: Text chunking and summarization using Groq.
   - `fact_checker.py`: Validates claims and conducts automated DuckDuckGo searches.
   - `tone_analyzer.py`: Identifies sentiment.
   - `reviewer.py`: Extracts quotes and audits grammar/bias.
3. **`src/workflow/graph.py`**: The `StateGraph` that manages parallel execution based on the router's output.
4. **`src/models/state.py`**: Pydantic models and TypedDicts defining the strongly-typed state schema.

---

## 🛠️ Technology Choices & Justification

- **LangGraph**: Crucial for the dynamic, parallel branching of the workflow. The `categorize_user_input` node outputs a list of actions, and LangGraph's conditional edges seamlessly route to one or multiple nodes simultaneously based on that list.
- **Groq (`llama-3.1-70b-versatile`)**: Replaced OpenAI for faster inference and open-source model usage. The speed is critical when analyzing multiple text chunks across 5 different agents concurrently.
- **DuckDuckGo Search**: Provides an unauthenticated, accessible way to execute programmatic web searches for fact-checking without needing dedicated Search API keys.
- **PyMuPDFLoader**: Used to parse sample PDF articles robustly.

---

## 5. Complete Agent Workflow

### Node Execution Flow

```text
┌──────────────────────────────────────────────────────────────────┐
│                        INPUTS                                     │
│  ┌─────────────────────┐    ┌──────────────────────────────┐      │
│  │    User Query       │    │     Article Text (PDF)       │      │
│  │ (e.g. "Full Report")│    │                              │      │
│  └────────┬────────────┘    └──────────────┬───────────────┘      │
│           │                                │                      │
│           └────────────┬───────────────────┘                      │
│                        ▼                                          │
│              ┌──────────────────┐                                 │
│              │     main.py      │    ← CLI Entry Point            │
│              │  (Orchestrator)  │                                 │
│              └────────┬─────────┘                                 │
│                       ▼                                           │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                  LangGraph Workflow (graph.py)               │  │
│  │                                                              │  │
│  │                  ┌───────────────────┐                       │  │
│  │                  │Categorize Input   │                       │  │
│  │                  │(Intent Router)    │                       │  │
│  │                  └─────────┬─────────┘                       │  │
│  │                            │                                 │  │
│  │      ┌─────────┬───────────┼───────────┬─────────┐           │  │
│  │      │         │           │           │         │           │  │
│  │      ▼         ▼           ▼           ▼         ▼           │  │
│  │  ┌───────┐ ┌───────┐   ┌───────┐   ┌───────┐ ┌───────┐       │  │
│  │  │Summ-  │ │Fact-  │   │Tone   │   │Quote  │ │Grammar│       │  │
│  │  │arize  │ │Check  │   │Analyze│   │Extract│ │& Bias │       │  │
│  │  └─┬─────┘ └─┬─────┘   └─┬─────┘   └─┬─────┘ └─┬─────┘       │  │
│  │    │         │           │           │         │             │  │
│  │    └─────────┴───────────┼───────────┴─────────┘             │  │
│  │                          ▼                                   │  │
│  │                     ┌─────────┐                              │  │
│  │                     │   END   │                              │  │
│  │                     └─────────┘                              │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                       │                                           │
│                       ▼                                           │
│              ┌──────────────────┐                                 │
│              │ Terminal Output  │    ← Formatted CLI Results      │
│              └──────────────────┘                                 │
└──────────────────────────────────────────────────────────────────┘
```

1. **User Request:** The user runs `python main.py` which loads the `Sample AI Generated Article.pdf`. The user types a request.
2. **Intent Routing:** The `categorize_user_input` node uses an LLM to map the query to a list of valid actions (e.g., `['summarization', 'fact-checking']`).
3. **Parallel Execution:** LangGraph conditionally routes the state to the requested nodes.
4. **Agent Processing:**
   - **Summarizer:** Chunks text and condenses it.
   - **Fact-Checker:** Extracts claims and searches DDG to verify them.
   - **Tone Analyzer:** Detects article sentiment.
   - **Reviewer:** Extracts quotes and flags grammatical or biased phrasing.
5. **Output:** The workflow reaches `END` and `main.py` prints the populated state back to the user.

---

## 🚀 How to Run

1. **Ensure API Keys are set** in your root `.env` file (`GROK_API_KEY`).
2. Navigate to the project directory:
   ```bash
   cd Langgraph/projects/020_journalism_agent
   ```
3. Install dependencies (if you haven't already):
   ```bash
   pip install duckduckgo-search langchain-groq pymupdf beautifulsoup4
   ```
4. Run the script:
   ```bash
   python3 main.py
   ```
5. Enter your prompt when requested.

---

## 📄 Resume Summary

### 3-Liner Summary
> - Architected a modular AI Journalism Assistant utilizing LangGraph and Groq LLMs to automatically chunk, summarize, fact-check, and review long-form articles for grammar and bias.
> - Integrated DuckDuckGo programmatic search pipelines to dynamically verify extracted claims in real-time, outputting structured JSON verification reports.
> - Engineered a parallel-branching state graph architecture capable of executing up to five distinct NLP analysis agents simultaneously based on intent-classified user inputs.

## 7. Interview Explanation Version

Journalism faces massive challenges with the proliferation of online misinformation, subtle biases, and information overload. Reporters often have to manually sift through enormous articles, verify obscure claims, and meticulously check their own text for unintentional bias—a time-consuming process under tight deadlines. To solve this, I architected a modular AI Journalism Assistant that acts as an automated editorial pipeline.

Technically, I engineered a parallel-branching state graph using LangGraph. When a user submits an article, an Intent Classifier categorizes the request and dynamically routes the text to up to five specialized agents simultaneously. The most critical component is the Fact-Checking Agent, which automatically extracts claims and validates them in real-time using a programmatic DuckDuckGo search integration, while parallel agents handle tone analysis, quote extraction, and summarization using high-speed Groq LLMs.

The business impact is a massively streamlined editorial workflow that guarantees journalistic integrity. By running articles through this automated pipeline, editorial teams can instantly verify claims against external evidence and neutralize unintentional bias in seconds, freeing up reporters to focus entirely on investigating and writing rather than tedious manual editing.
