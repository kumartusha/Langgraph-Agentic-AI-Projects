from typing import TypedDict, Annotated, List, Dict
from pydantic import BaseModel, Field

class GraphState(TypedDict):
    """
    Represents the state of our News Summarization agent.
    """
    news_query: Annotated[str, "Input query to extract news search parameters from."]
    num_searches_remaining: Annotated[int, "Number of articles to search for."]
    newsapi_params: Annotated[dict, "Structured argument for the News API."]
    past_searches: Annotated[List[dict], "List of search params already used."]
    articles_metadata: Annotated[List[dict], "Article metadata response from the News API"]
    scraped_urls: Annotated[List[str], "List of urls already scraped."]
    num_articles_tldr: Annotated[int, "Number of articles to create TL;DR for."]
    potential_articles: Annotated[List[Dict[str, str]], "Article with full text to consider summarizing."]
    tldr_articles: Annotated[List[Dict[str, str]], "Selected article TL;DRs."]
    formatted_results: Annotated[str, "Formatted results to display."]

class NewsApiParams(BaseModel):
    """
    Schema to force the LLM to output valid NewsAPI parameters.
    """
    q: str = Field(description="1-3 concise keyword search terms that are not too specific")
    sources: str = Field(description="comma-separated list of sources from: 'abc-news,bbc-news,techcrunch,bloomberg,business-insider,cnn,fortune'")
    from_param: str = Field(description="date in format 'YYYY-MM-DD' Two days ago minimum. Extend up to 30 days on second and subsequent requests.", alias="from")
    to: str = Field(description="date in format 'YYYY-MM-DD' today's date unless specified")
    language: str = Field(description="language of articles 'en' unless specified")
    sort_by: str = Field(description="sort by 'relevancy', 'popularity', or 'publishedAt'")

    class Config:
        populate_by_name = True
