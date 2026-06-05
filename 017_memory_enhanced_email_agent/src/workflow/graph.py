from langgraph.graph import StateGraph, START, END
from langgraph.store.memory import InMemoryStore
from src.models.state import State
from src.agents.triage import triage_email_node
from src.agents.responder import create_responder_agent

def route_based_on_triage(state: State):
    if state.get("triage_result") == "respond":
        return "response_agent"
    return END

def create_email_workflow(store: InMemoryStore):
    # Define the workflow graph
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("triage", lambda state, config: triage_email_node(state, config, store))
    
    # Create the response agent that has the tools and dynamic prompts
    response_agent = create_responder_agent(store)
    workflow.add_node("response_agent", response_agent)
    
    # Set edges
    workflow.add_edge(START, "triage")
    workflow.add_conditional_edges("triage", route_based_on_triage, {
        "response_agent": "response_agent",
        END: END
    })
    
    # Compile the graph passing the memory store
    return workflow.compile(store=store)
