import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import AgentState
from pydantic import BaseModel, Field

dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.env"))
load_dotenv(dotenv_path)

class ReconciliationResult(BaseModel):
    flags: list[str] = Field(description="List of medication discrepancies or warnings.")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
structured_llm = llm.with_structured_output(ReconciliationResult)

def medication_reconciliation_node(state: AgentState) -> dict:
    """
    LangGraph node to compare extracted medications with ground truth admission medications.
    """
    print("---RUNNING MEDICATION RECONCILIATION AGENT---")
    extracted_meds = [m.get("name", "") for m in state.get("extracted_medications", [])]
    ground_truth_meds = state.get("ground_truth_meds", [])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a clinical pharmacist agent. Compare the 'Extracted Medications' (found in the discharge note) against the 'Admission Medications' (ground truth database). Flag any discrepancies (e.g., a medication that was on admission but missing from discharge, or vice-versa). Be concise. If there are no discrepancies and they match perfectly, return an empty list."),
        ("human", "Admission Medications:\n{ground_truth}\n\nExtracted Medications (Discharge):\n{extracted}")
    ])
    
    chain = prompt | structured_llm
    result = chain.invoke({"ground_truth": ground_truth_meds, "extracted": extracted_meds})
    
    return {"reconciliation_flags": result.flags}
