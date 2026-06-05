# Visual Architecture & System Diagrams

This document focuses entirely on the structural and visual representation of the **Agentic Patient Discharge Summary Generator**.

## 1. High-Level Enterprise Architecture (Full Spec)
*This diagram represents the complete production environment, including AWS deployment layers, data ingestion, the AI processing core, and final export.*

```mermaid
graph TB
    subgraph "Data Ingestion Layer"
        S3[(AWS S3\nRaw Data)]
        PDF[PDF Uploads]
        HL7[HL7 / CSV Feeds]
        S3 -.-> PDF & HL7
    end

    subgraph "Compute & Orchestration Layer (AWS ECS Fargate)"
        subgraph "LangGraph Core"
            Parser[Document Parser Agent]
            NER[Medical NER Agent]
            Recon[Reconciler Agent]
            Draft[Drafting Agent]
            Safe[Safety Agent]
            
            Parser -->|Extracted Text| NER
            NER -->|Entities| Recon
            Recon -->|Flags| Draft
            Draft -->|Draft| Safe
        end
    end

    subgraph "Human-in-the-Loop & Export Layer"
        UI{{Streamlit Physician UI}}
        Cognito((AWS Cognito Auth))
        DB[(PostgreSQL\nAudit DB)]
        FHIR[[AWS HealthLake\nFHIR Export]]
    end

    PDF & HL7 --> Parser
    Safe -->|If Flags/Issues| UI
    Safe -->|If Approved| FHIR
    UI -->|Manual Override| FHIR
    UI -.-> Cognito
    LangGraph Core -.->|Audit Logs| DB

    %% Styling
    classDef aws fill:#FF9900,stroke:#232F3E,stroke-width:2px,color:white;
    classDef ai fill:#00A4A6,stroke:#005555,stroke-width:2px,color:white;
    classDef data fill:#3F88C5,stroke:#1A3A53,stroke-width:2px,color:white;
    
    class S3,Cognito,FHIR aws;
    class Parser,NER,Recon,Draft,Safe ai;
    class PDF,HL7,DB,UI data;
```

---

## 2. Core LangGraph State Machine (Current MVP)
*This details the exact Directed Acyclic Graph (DAG) executed in `src/graph.py` and how the memory (`AgentState`) is updated at each node.*

```mermaid
stateDiagram-v2
    direction TB
    
    state "Input Processing" as Input
    state "NER Agent Node" as Node1
    state "Reconciler Agent Node" as Node2
    state "Drafting Agent Node" as Node3
    state "Safety Agent Node" as Node4
    
    [*] --> Input : Loads NOTEEVENTS.csv & PRESCRIPTIONS.csv
    
    Input --> Node1 : Initial AgentState
    note right of Input
      {
        raw_clinical_note: str,
        ground_truth_meds: list
      }
    end note
    
    Node1 --> Node2 : + extracted_entities
    note right of Node1
      Extracts structured ICD-10 
      diagnoses & medications via LLM
    end note
    
    Node2 --> Node3 : + reconciliation_flags
    note right of Node2
      Cross-references NER meds 
      with ground truth (Python Logic)
    end note
    
    Node3 --> Node4 : + draft_summary
    note right of Node3
      LLM writes SOAP note using
      raw text + flags
    end note
    
    Node4 --> [*] : + safety_approved, safety_feedback
    note left of Node4
      LLM-as-a-Judge verifies 
      no harmful hallucinations
    end note
```

---

## 3. Data Flow & State Evolution (Sequence Diagram)
*This visualizes the chronological communication between the Orchestrator, the LLMs, and the shared `AgentState`.*

```mermaid
sequenceDiagram
    autonumber
    actor User/System
    participant Orchestrator as LangGraph Orchestrator
    participant State as Shared AgentState
    participant LLM as Claude/Llama3
    
    User/System->>Orchestrator: Invoke Graph (Note, Prescriptions)
    Orchestrator->>State: Initialize State
    
    rect rgb(235, 245, 255)
        Note right of Orchestrator: NER Phase
        Orchestrator->>LLM: Prompt + raw_note + Schema
        LLM-->>Orchestrator: Structured Entities JSON
        Orchestrator->>State: Update `extracted_entities`
    end
    
    rect rgb(255, 245, 235)
        Note right of Orchestrator: Reconciliation Phase (No LLM)
        Orchestrator->>State: Read `ground_truth` & `extracted`
        Orchestrator->>State: Write Discrepancies to `reconciliation_flags`
    end
    
    rect rgb(235, 255, 235)
        Note right of Orchestrator: Drafting Phase
        Orchestrator->>State: Read Note, Entities, Flags
        Orchestrator->>LLM: Prompt + All Context
        LLM-->>Orchestrator: Generated SOAP Markdown
        Orchestrator->>State: Update `draft_summary`
    end
    
    rect rgb(255, 235, 235)
        Note right of Orchestrator: Safety Phase
        Orchestrator->>State: Read Note & Draft
        Orchestrator->>LLM: Evaluation Prompt (LlamaGuard sim)
        LLM-->>Orchestrator: Boolean Approval & Feedback
        Orchestrator->>State: Update `safety` keys
    end
    
    Orchestrator-->>User/System: Return Final State Dictionary
```

---

## 4. Medication Reconciliation Logic Tree
*A deep dive into how `src/agents/reconciler.py` determines flags without using generative AI.*

```mermaid
flowchart TD
    Start((Start Reconciler)) --> Load[Load `extracted_meds` & `ground_truth_meds`]
    Load --> Loop[For each med in extracted_meds]
    
    Loop --> Check{Is Med in <br> ground_truth?}
    
    Check -- Yes --> Match[Match Found: Safe]
    Check -- No --> Flag1[/Flag: Medication mentioned in note but NOT prescribed/]
    
    Loop2[For each med in ground_truth] --> Check2{Is Med in <br> extracted_meds?}
    Load --> Loop2
    
    Check2 -- Yes --> Match
    Check2 -- No --> Flag2[/Flag: Medication prescribed but MISSING from discharge note/]
    
    Flag1 --> Append[Append to `reconciliation_flags`]
    Flag2 --> Append
    
    Append --> End((Return Updated State))
    Match --> End
```

---

## 5. Pydantic Data Models (Class Diagram)
*This shows the strict data contracts defined in `src/models/schemas.py` that force the LLM to output predictable JSON.*

```mermaid
classDiagram
    class AgentState {
        <<TypedDict>>
        +String raw_clinical_note
        +List[String] ground_truth_meds
        +DischargeSummary extracted_entities
        +List[String] reconciliation_flags
        +String draft_summary
        +Boolean safety_approved
        +String safety_feedback
    }

    class DischargeSummary {
        <<Pydantic Model>>
        +List[Diagnosis] diagnoses
        +List[Medication] medications
        +List[String] procedures
    }

    class Diagnosis {
        <<Pydantic Model>>
        +String condition_name
        +String icd10_code_guess
        +Float confidence_score
    }

    class Medication {
        <<Pydantic Model>>
        +String drug_name
        +String dosage
        +String frequency
    }

    AgentState *-- DischargeSummary : contains
    DischargeSummary *-- Diagnosis : contains
    DischargeSummary *-- Medication : contains
```

---

## 6. Codebase File Dependency Graph
*Visualizing how the Python files interact and import each other in the `src/` directory.*

```mermaid
graph TD
    subgraph Data Layer
        CSV1[data/NOTEEVENTS.csv]
        CSV2[data/PRESCRIPTIONS.csv]
    end

    subgraph Entry Point
        Main[src/main.py]
    end

    subgraph Orchestrator
        Graph[src/graph.py]
        State[src/state.py]
    end

    subgraph Agent Nodes
        NER[src/agents/ner.py]
        Recon[src/agents/reconciler.py]
        Draft[src/agents/drafter.py]
        Safety[src/agents/safety.py]
    end

    subgraph Data Models
        Schemas[src/models/schemas.py]
    end

    CSV1 & CSV2 -.->|Parsed by| Main
    Main -->|Invokes| Graph
    Graph -->|Uses| State
    Graph -->|Routes to| NER & Recon & Draft & Safety
    
    NER -->|Validates via| Schemas
    Draft -->|Validates via| Schemas
    Safety -->|Validates via| Schemas
    State -->|Types via| Schemas
```
