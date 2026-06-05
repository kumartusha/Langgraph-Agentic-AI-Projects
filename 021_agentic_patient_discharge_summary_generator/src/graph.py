from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.agents.ner import medical_ner_node
from src.agents.reconciler import medication_reconciliation_node
from src.agents.drafter import drafting_node
from src.agents.safety import safety_node

def build_graph():
    """Builds and compiles the LangGraph state machine."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("NER_Agent", medical_ner_node)
    workflow.add_node("Reconciler_Agent", medication_reconciliation_node)
    workflow.add_node("Drafting_Agent", drafting_node)
    workflow.add_node("Safety_Agent", safety_node)
    
    # Define edges (linear flow)
    workflow.set_entry_point("NER_Agent")
    workflow.add_edge("NER_Agent", "Reconciler_Agent")
    workflow.add_edge("Reconciler_Agent", "Drafting_Agent")
    workflow.add_edge("Drafting_Agent", "Safety_Agent")
    workflow.add_edge("Safety_Agent", END)
    
    # Compile
    app = workflow.compile()
    return app
