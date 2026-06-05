from langgraph.graph import StateGraph, START, END

from src.models.state import GraphState
from src.agents.searcher import search_node
from src.agents.summarizer import summarize_node
from src.agents.publisher import publish_node

def create_workflow() -> StateGraph:
    """
    Constructs and configures the workflow graph
    search -> summarize -> publish
    
    Returns:
        StateGraph: Compiled workflow ready for execution
    """
    workflow = StateGraph(GraphState)
    
    workflow.add_node("search", search_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("publish", publish_node)
    
    workflow.add_edge(START, "search")
    workflow.add_edge("search", "summarize") 
    workflow.add_edge("summarize", "publish")
    workflow.add_edge("publish", END)
    
    return workflow.compile()
