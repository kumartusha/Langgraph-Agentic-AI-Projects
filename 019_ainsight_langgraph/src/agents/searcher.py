from typing import List, Dict, Any
from tavily import TavilyClient

from src.config.settings import settings
from src.models.state import Article, GraphState

class NewsSearcher:
    """
    Agent responsible for finding relevant AI/ML news articles
    using the Tavily search API
    """
    
    def __init__(self):
        self.tavily = TavilyClient(api_key=settings.tavily_api_key)
        
    def search(self) -> List[Article]:
        """
        Performs news search with configured parameters
        
        Returns:
            List[Article]: Collection of found articles
        """
        response = self.tavily.search(
            query="artificial intelligence and machine learning news", 
            topic="news",
            time_period="1w",
            search_depth="advanced",
            max_results=5
        )
        
        articles = []
        for result in response['results']:
            articles.append(Article(
                title=result['title'],
                url=result['url'],
                content=result['content']
            ))
        
        return articles

def search_node(state: GraphState) -> GraphState:
    """
    Node for article search
    
    Args:
        state (GraphState): Current workflow state
        
    Returns:
        GraphState: Updated state with found articles
    """
    searcher = NewsSearcher()
    state['articles'] = searcher.search() 
    return state
