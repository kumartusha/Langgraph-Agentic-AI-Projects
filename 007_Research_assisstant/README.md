# AI Academic Research Assistant (LangGraph + Groq)

## Problem Statement
In the modern academic landscape, researchers are faced with a massive information overload. When starting a new project or exploring a novel domain, a researcher must query vast databases, download dozens of PDFs, manually read through them, and synthesize the findings. This manual literature review process is incredibly time-consuming, prone to human error, and creates a significant bottleneck in scientific discovery. 

## The Solution
This project implements a fully autonomous, stateful **AI Research Assistant** utilizing **LangChain**, **LangGraph**, and the lightning-fast **Groq API**. 

Instead of a simple "chat" interface, this assistant is modeled as a cyclic graph (state machine) that possesses agency. It can independently decide when to search for papers, dynamically download and read PDFs, and most importantly, critically evaluate its own work to ensure the final output is highly accurate and deeply cited before presenting it to the user.

---

## Detailed Architecture & Flow

The system is built on **LangGraph**, consisting of several distinct "nodes" that represent the cognitive steps of a real human researcher.

### 1. Decision Making Node
- **Purpose:** Acts as the entry gatekeeper. 
- **Flow:** When the user inputs a query, this node determines if the question is a simple conversational greeting (e.g., "Hello") or a complex query requiring research.
- **Outcome:** If conversational, it answers directly. If research is needed, it routes the workflow to the **Planning Node**.

### 2. Planning Node
- **Purpose:** Strategy formulation.
- **Flow:** The LLM takes the user's research objective and creates a step-by-step execution plan. It looks at the available tools and maps out exactly what needs to be searched and analyzed.

### 3. Agent & Tools Loop
- **Purpose:** The execution engine.
- **Flow:** The agent begins calling external tools based on its plan.
  - **Tool 1 (`search_papers`):** Hooks into the **CORE API** to query millions of open-access scientific papers. It leverages complex queries (booleans, date ranges) to find highly relevant literature.
  - **Tool 2 (`download_paper`):** Once a relevant paper URL is found, this tool fetches the PDF, mimics a real browser to bypass generic firewalls, and extracts the raw text using `pdfplumber`.
  - **Tool 3 (`ask_human_feedback`):** A fallback mechanism for unexpected errors.
- **Outcome:** The agent compiles all the raw data, synthesizes it, and drafts a final response rich with inline citations.

### 4. Judge Node
- **Purpose:** Quality Assurance (Self-Reflection).
- **Flow:** Before the user sees the answer, the Judge LLM reviews the drafted response. It checks if the answer directly addresses the prompt, if it is extensive enough, and if it includes proper inline citations.
- **Outcome:** If the answer is satisfactory, it is presented to the user. If the answer is lacking (e.g., missing citations), the Judge provides strict feedback and routes the workflow *back* to the **Planning Node** to try again. (A safety limit of 2 retries prevents infinite loops).

---

## Technical Stack
- **Orchestration:** LangGraph (Stateful Cyclic Graphs) & LangChain
- **LLM Provider:** Groq (`qwen/qwen3-32b` / `llama3-70b-8192` for high-speed function calling)
- **Data Sources:** CORE API (Open Access Research Papers)
- **PDF Parsing:** `pdfplumber`

---

## How to Run

### 1. Install Dependencies
Ensure you have Python 3 installed, then install the required libraries:
```bash
pip3 install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory (or update the existing one) with your API keys:
```env
CORE_API_KEY=your_core_api_key_here
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 3. Execute the CLI
Run the assistant directly from your terminal using the Python module syntax:
```bash
python3 -m src.main --query "What are the latest advancements in quantum machine learning?"
```
The console will print out beautifully colored logs showing you the exact node the agent is currently thinking in, followed by the final thoroughly researched answer.
