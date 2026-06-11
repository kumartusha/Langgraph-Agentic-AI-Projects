# 🏥 Clinical Discharge Summary Generator — Complete Workflow Documentation

## 📌 Project Overview

A **production-grade, multi-agent AI pipeline** built using **LangGraph** that automates clinical
discharge summary generation from raw hospital notes (MIMIC-III format). The system orchestrates
4 specialized AI agents in a structured DAG (Directed Acyclic Graph), each responsible for a
distinct cognitive task — from entity extraction to safety validation — and outputs a final
professional **SOAP-format PDF report**.

---

## 🏗️ System Architecture

```
RAW CLINICAL NOTE
      │
      ▼
┌─────────────────────┐
│   NER Agent         │  ◄── Groq LLM (llama3-70b)
│  (Entity Extract)   │       Pulls: diagnoses, meds, dosages from raw text
└────────┬────────────┘
         │  AgentState (entities + meds list)
         ▼
┌─────────────────────┐
│  Reconciler Agent   │  ◄── Deterministic Python Logic (no LLM)
│  (Medication Check) │       Compares: extracted meds vs. ground truth
└────────┬────────────┘
         │  AgentState (+ reconciliation_flags)
         ▼
┌─────────────────────┐
│  Drafting Agent     │  ◄── Groq LLM (llama3-70b)
│  (SOAP Generator)   │       Writes: Subjective, Objective, Assessment, Plan
└────────┬────────────┘
         │  AgentState (+ draft_summary)
         ▼
┌─────────────────────┐
│  Safety Agent       │  ◄── Groq LLM (llama3-70b) acting as Judge
│  (LLM-as-a-Judge)  │       Evaluates: hallucinations, omissions, drug errors
└────────┬────────────┘
         │  AgentState (+ safety_approved, safety_feedback)
         ▼
┌─────────────────────────────────────────┐
│  OUTPUT                                 │
│  ├── Rich Terminal (colored, streaming) │
│  └── PDF Report (discharge_summary.pdf) │
└─────────────────────────────────────────┘
```

---

## 🔁 Execution Workflow — Step by Step

### Step 0: Data Loading (`main.py`)
- Reads synthetic MIMIC-III style data from `data/NOTEEVENTS.csv` and `data/PRESCRIPTIONS.csv`
- Loads a single patient record (Subject ID + HADM ID)
- Extracts the raw clinical note text and the ground-truth medication list
- Loads the GROQ_API_KEY from the parent `.env` file using `python-dotenv`

### Step 1: Graph Construction (`src/graph.py`)
- Instantiates a `StateGraph(AgentState)` — LangGraph's core DAG engine
- Registers 4 nodes: `NER_Agent`, `Reconciler_Agent`, `Drafting_Agent`, `Safety_Agent`
- Adds directed edges forming a **linear DAG**: NER → Reconciler → Drafter → Safety → END
- Compiles the graph into an executable `app` object

### Step 2: NER Agent (`src/agents/ner.py`)
- **Trigger:** Receives `raw_clinical_note` from the shared `AgentState`
- **Action:** Sends the raw text to Groq LLM with a structured prompt requesting JSON output
- **Output written to state:**
  - `extracted_medications` — list of all drug names found
  - `extracted_diagnoses` — list of conditions identified
  - `extracted_entities` — all named medical entities (comprehensive)
- **Technology:** `ChatGroq` via `langchain-groq`, forced structured JSON output via prompt engineering

### Step 3: Reconciler Agent (`src/agents/reconciler.py`)
- **Trigger:** Receives `extracted_medications` + `ground_truth_meds` from state
- **Action:** Runs pure Python comparison logic — NO LLM involved
  - Checks for medications in ground truth that are missing from extracted meds
  - Flags duplicate medications in the admission record
  - Detects formulation discrepancies (e.g., "Alteplase" vs. "Alteplase (Catheter Clearance)")
- **Output written to state:**
  - `discharge_medications` — normalized list of discharge meds
  - `reconciliation_flags` — list of human-readable flag strings
- **Design Choice:** Deterministic logic here ensures 100% reproducible, auditable medication checks — no hallucination risk from an LLM

### Step 4: Drafting Agent (`src/agents/drafter.py`)
- **Trigger:** Receives `extracted_diagnoses`, `discharge_medications`, `reconciliation_flags`
- **Action:** Calls Groq LLM with a clinical writing prompt to generate a **full SOAP note**
- **SOAP Sections Generated:**
  - **S (Subjective):** Patient's reported symptoms and history
  - **O (Objective):** Vital signs, lab results, medication list
  - **A (Assessment):** Clinical diagnoses summary
  - **P (Plan):** Discharge medications, follow-up instructions, flags
- **Output written to state:** `draft_summary` — the complete SOAP note as a string

### Step 5: Safety Agent (`src/agents/safety.py`)
- **Trigger:** Receives `draft_summary` + `reconciliation_flags` + `ground_truth_meds`
- **Action:** Acts as an **LLM-as-a-Judge** — sends the draft to Groq LLM with a strict evaluation rubric
- **Evaluates:**
  - Are all critical medications listed in the discharge plan?
  - Are there any dosage errors or ambiguities?
  - Does the summary contain hallucinated information not present in the original note?
  - Are reconciliation flags addressed in the plan?
- **Output written to state:**
  - `safety_approved` (bool) — True only if no critical issues found
  - `safety_feedback` (str) — Detailed reasoning from the judge LLM
- **Design Choice:** A second LLM "judge" evaluates the first LLM's output — classic **self-critique / Constitutional AI** pattern

### Step 6: Streaming Output Display (`main.py`)
- Uses `app.stream(inputs)` instead of `app.invoke()` for **real-time, event-driven output**
- Each agent's completion event triggers a live terminal print:
  ```
  [1/4] Medical NER     DONE  Extracting entities from clinical notes...
  [2/4] Reconciler      DONE  Comparing meds against ground truth...
  [3/4] SOAP Drafter    DONE  Generating discharge summary...
  [4/4] Safety Judge    DONE  Running safety & hallucination check...
  ```
- Uses Python `rich` library for colored panels, tables, and rule separators

### Step 7: PDF Report Generation (`main.py → ClinicalPDF class`)
- Custom `ClinicalPDF` class extends `fpdf2.FPDF`
- **Sections generated in the PDF:**
  1. **Navy Blue Header Banner** — "MediSys AI | Clinical Discharge Report"
  2. **Patient & Admission Details** — Subject ID, HADM ID, timestamp, pipeline info
  3. **Admitted Medication List** — Ground-truth meds in 2-column numbered layout
  4. **AI-Generated SOAP Note** — Color-coded by section (Subjective=blue, Objective=green, Assessment=orange, Plan=purple)
  5. **Medication Reconciliation Flags** — Red bullet-point list
  6. **AI Safety Assessment** — Green (approved) or Red (rejected) status block
  7. **Legal Disclaimer** — Yellow advisory box
  8. **Footer** — Page number, timestamp, CONFIDENTIAL label
- **Unicode Sanitization:** All LLM output is passed through a `sanitize()` function that replaces non-Latin-1 characters (em dashes, curly quotes, arrows) with ASCII equivalents before hitting the PDF engine

---

## 🗂️ Project File Structure

```
021_Clinical_discharge_summary_generator/
├── src/
│   ├── graph.py              ← LangGraph DAG builder
│   ├── state.py              ← AgentState TypedDict definition
│   ├── main.py               ← Entry point: orchestrates run + PDF + terminal output
│   ├── agents/
│   │   ├── ner.py            ← Medical NER agent (LLM)
│   │   ├── reconciler.py     ← Medication reconciliation (deterministic Python)
│   │   ├── drafter.py        ← SOAP note drafting agent (LLM)
│   │   └── safety.py         ← Safety evaluation agent (LLM-as-Judge)
│   └── models/               ← Pydantic schemas for structured data contracts
├── data/
│   ├── NOTEEVENTS.csv        ← Synthetic MIMIC-III clinical notes
│   └── PRESCRIPTIONS.csv     ← Synthetic medication prescriptions
├── output/
│   └── discharge_summary.pdf ← Generated PDF report
├── generate_synthetic_notes.py ← Script to generate synthetic test data
├── requirements.txt
└── resourcees/
    ├── workflow_documentation.md   ← THIS FILE
    └── interview_qa.md             ← Interview Q&A guide
```

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Orchestration | **LangGraph** (`StateGraph`) | Multi-agent DAG execution engine |
| LLM Provider | **Groq** (`llama3-70b-8192`) | Fast inference for NER, Drafting, Safety |
| LLM Framework | **LangChain** (`langchain-groq`, `langchain-core`) | LLM abstraction + prompt management |
| Data Modeling | **Pydantic** | Structured output validation and state contracts |
| Data Processing | **Pandas** | CSV loading and patient record filtering |
| Terminal Output | **Rich** | Colorful panels, tables, rules, live streaming |
| PDF Generation | **fpdf2** | Professional PDF report creation |
| Config | **python-dotenv** | Secure API key management via `.env` |
| Data Source | **MIMIC-III Demo** (Synthetic) | Clinical note and prescription data |

---

## ▶️ How to Run

```bash
# 1. Install dependencies
py -m pip install -r requirements.txt

# 2. Set up your API key in the parent .env file
# GROQ_API_KEY="your_key_here"

# 3. Generate synthetic clinical data
py generate_synthetic_notes.py

# 4. Run the full pipeline
py -m src.main
```

Output PDF will be saved to `output/discharge_summary.pdf`.
