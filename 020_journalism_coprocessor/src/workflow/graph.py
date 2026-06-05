from langgraph.graph import StateGraph, END, START
from src.models.state import State
from src.agents.router import categorize_user_input
from src.agents.summarizer import summarization_node
from src.agents.fact_checker import fact_checking_node
from src.agents.tone_analyzer import tone_analysis_node
from src.agents.reviewer import quote_extraction_node, grammar_and_bias_review_node

# Define constants
CATEGORY = "categorize_user_input"
SUMMARY = "summarization"
FACT_CHECKING = "fact-checking"
TONE_ANALYSIS = "tone-analysis"
QUOTE_EXTRACTION = "quote-extraction"
GRAMMAR_AND_BIAS_REVIEW = "grammar-and-bias-review"

def route(state: State):
    routes = []
    actions = state.get("actions", [])
    
    valid_routes = [SUMMARY, FACT_CHECKING, TONE_ANALYSIS, QUOTE_EXTRACTION, GRAMMAR_AND_BIAS_REVIEW]
    
    for action in actions:
        if action in valid_routes:
            routes.append(action)

    if not routes:
        return END

    return routes

def create_workflow() -> StateGraph:
    workflow = StateGraph(State)

    # Define the nodes
    workflow.add_node(CATEGORY, categorize_user_input)
    workflow.add_node(SUMMARY, summarization_node)
    workflow.add_node(FACT_CHECKING, fact_checking_node)
    workflow.add_node(TONE_ANALYSIS, tone_analysis_node)
    workflow.add_node(QUOTE_EXTRACTION, quote_extraction_node)
    workflow.add_node(GRAMMAR_AND_BIAS_REVIEW, grammar_and_bias_review_node)

    # Start at categorization
    workflow.add_edge(START, CATEGORY)

    # Conditional routing to the different tools
    workflow.add_conditional_edges(
        CATEGORY,
        route,
        {
            SUMMARY: SUMMARY,
            FACT_CHECKING: FACT_CHECKING,
            TONE_ANALYSIS: TONE_ANALYSIS,
            QUOTE_EXTRACTION: QUOTE_EXTRACTION,
            GRAMMAR_AND_BIAS_REVIEW: GRAMMAR_AND_BIAS_REVIEW,
            END: END,
        }
    )

    # All tools end the graph once they are done
    workflow.add_edge(SUMMARY, END)
    workflow.add_edge(FACT_CHECKING, END)
    workflow.add_edge(TONE_ANALYSIS, END)
    workflow.add_edge(QUOTE_EXTRACTION, END)
    workflow.add_edge(GRAMMAR_AND_BIAS_REVIEW, END)

    return workflow.compile()
