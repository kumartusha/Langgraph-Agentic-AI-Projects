from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.models.state import GraphState, NewsApiParams
from src.config.settings import settings

def generate_newsapi_params(state: GraphState) -> GraphState:
    """
    Based on the natural language query, generate strict News API JSON params.
    """
    llm = ChatGroq(model=settings.groq_model, api_key=settings.grok_api_key)
    parser = JsonOutputParser(pydantic_object=NewsApiParams)

    today_date = datetime.now().strftime("%Y-%m-%d")
    past_searches = state.get("past_searches", [])
    num_searches_remaining = state.get("num_searches_remaining", 10)
    news_query = state.get("news_query", "")

    template = """
    Today is {today_date}.

    Create a param dict for the News API based on the user query:
    {query}

    These searches have already been made. Loosen the search terms to get more results.
    {past_searches}
    
    Following these formatting instructions:
    {format_instructions}

    Including this one, you have {num_searches_remaining} searches remaining.
    If this is your last search, use all news sources and a 30 days search range.
    """

    prompt_template = PromptTemplate(
        template=template,
        variables={
            "today_date": today_date,
            "query": news_query,
            "past_searches": past_searches,
            "num_searches_remaining": num_searches_remaining
        },
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt_template | llm | parser

    result = chain.invoke({
        "query": news_query, 
        "today_date": today_date, 
        "past_searches": past_searches, 
        "num_searches_remaining": num_searches_remaining
    })

    # The schema uses 'from_param' to avoid python 'from' keyword, but NewsAPI needs 'from'
    if "from_param" in result:
        result["from"] = result.pop("from_param")

    state["newsapi_params"] = result
    return state
