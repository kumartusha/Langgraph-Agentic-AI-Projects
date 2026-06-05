from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    """State object that passes through the LangGraph agents."""
    
    # Inputs
    raw_clinical_note: str
    ground_truth_meds: List[str]
    
    # NER Outputs (Parsed from Pydantic schemas)
    extracted_diagnoses: List[Dict[str, Any]]
    extracted_medications: List[Dict[str, Any]]
    extracted_procedures: List[str]
    
    # Reconciler Outputs
    reconciliation_flags: List[str]
    
    # Drafter Outputs
    draft_summary: str
    
    # Safety Outputs
    safety_approved: bool
    safety_feedback: str
