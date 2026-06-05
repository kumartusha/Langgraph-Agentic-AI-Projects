from langgraph.graph import StateGraph, START, END

from src.models.state import GraphState
from src.agents.query_generator import generate_newsapi_params
from src.agents.news_fetcher import retrieve_articles_metadata, retrieve_articles_text
from src.agents.summarizer import select_top_urls, summarize_articles_parallel, format_results

def articles_text_decision(state: GraphState) -> str:
    """
    Check results of retrieve_articles_text to determine next step.
    If we haven't found enough valid articles, loop back and query again with looser params.
    """
    if state.get("num_searches_remaining", 0) <= 0:
        if len(state.get("potential_articles", [])) == 0:
            state["formatted_results"] = "No articles with text found."
            return "END"
        else:
            return "select_top_urls"
    else:
        if len(state.get("potential_articles", [])) < state.get("num_articles_tldr", 3):
            return "generate_newsapi_params"
        else:
            return "select_top_urls"

def create_workflow():
    """
    Compile and return the LangGraph workflow.
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("generate_newsapi_params", generate_newsapi_params)
    workflow.add_node("retrieve_articles_metadata", retrieve_articles_metadata)
    workflow.add_node("retrieve_articles_text", retrieve_articles_text)
    workflow.add_node("select_top_urls", select_top_urls)
    workflow.add_node("summarize_articles_parallel", summarize_articles_parallel)
    workflow.add_node("format_results", format_results)

    workflow.add_edge(START, "generate_newsapi_params")
    workflow.add_edge("generate_newsapi_params", "retrieve_articles_metadata")
    workflow.add_edge("retrieve_articles_metadata", "retrieve_articles_text")
    
    workflow.add_conditional_edges(
        "retrieve_articles_text",
        articles_text_decision,
        {
            "generate_newsapi_params": "generate_newsapi_params",
            "select_top_urls": "select_top_urls",
            "END": END
        }
    )
    
    workflow.add_edge("select_top_urls", "summarize_articles_parallel")
    
    workflow.add_conditional_edges(
        "summarize_articles_parallel",
        lambda state: "format_results" if len(state.get("tldr_articles", [])) > 0 else "END",
        {
            "format_results": "format_results",
            "END": END
        }
    )
    
    workflow.add_edge("format_results", END)

    return workflow.compile()
