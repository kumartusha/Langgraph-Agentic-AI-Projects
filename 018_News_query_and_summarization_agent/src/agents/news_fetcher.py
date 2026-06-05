import requests
from bs4 import BeautifulSoup
from newsapi import NewsApiClient

from src.models.state import GraphState
from src.config.settings import settings

def retrieve_articles_metadata(state: GraphState) -> GraphState:
    """
    Using the structured NewsAPI params, perform API call.
    """
    newsapi_params = state["newsapi_params"]
    
    # Initialize past searches and scraped urls if empty
    if "past_searches" not in state:
        state["past_searches"] = []
    if "scraped_urls" not in state:
        state["scraped_urls"] = []
    if "potential_articles" not in state:
        state["potential_articles"] = []
    
    state['num_searches_remaining'] -= 1

    try:
        newsapi = NewsApiClient(api_key=settings.newsapi_key)
        
        # NewsAPI requires 'from_param' instead of 'from' in its python wrapper kwargs
        kwargs = newsapi_params.copy()
        if "from" in kwargs:
            kwargs["from_param"] = kwargs.pop("from")

        articles = newsapi.get_everything(**kwargs)
        
        # Remember this exact search so we don't repeat it
        state['past_searches'].append(newsapi_params)

        scraped_urls = state["scraped_urls"]
        new_articles = []
        
        for article in articles.get('articles', []):
            if article.get('url') not in scraped_urls and len(state['potential_articles']) + len(new_articles) < 10:
                new_articles.append(article)

        state["articles_metadata"] = new_articles
    except Exception as e:
        print(f"NewsAPI Error: {e}")
        state["articles_metadata"] = []

    return state

def retrieve_articles_text(state: GraphState) -> GraphState:
    """
    Web scrapes to retrieve the full article text from the URLs.
    """
    articles_metadata = state.get("articles_metadata", [])
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36'
    }

    potential_articles = []

    for article in articles_metadata:
        url = article.get('url')
        if not url:
            continue

        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text(strip=True)
                
                # Basic validation to ensure we didn't just scrape an empty page or paywall block
                if len(text) > 200:
                    potential_articles.append({
                        "title": article.get("title", "No Title"),
                        "url": url,
                        "description": article.get("description", ""),
                        "text": text
                    })
                    state["scraped_urls"].append(url)
        except requests.RequestException:
            pass # Ignore unreachable sites
            
    state["potential_articles"].extend(potential_articles)
    return state
