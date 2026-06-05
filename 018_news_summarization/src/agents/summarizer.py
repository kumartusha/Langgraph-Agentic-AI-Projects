import re
from langchain_groq import ChatGroq

from src.models.state import GraphState
from src.config.settings import settings

def select_top_urls(state: GraphState) -> GraphState:
    """
    Based on the article synopses, choose the top-n articles to summarize using an LLM.
    """
    llm = ChatGroq(model=settings.groq_model, api_key=settings.grok_api_key)
    
    news_query = state["news_query"]
    num_articles_tldr = state["num_articles_tldr"]
    potential_articles = state.get("potential_articles", [])

    formatted_metadata = "\n".join([f"{a['url']}\n{a['description']}\n" for a in potential_articles])

    prompt = f"""
    Based on the user news query:
    {news_query}

    Reply with a list of strings of up to {num_articles_tldr} relevant urls.
    Don't add any urls that are not relevant or aren't listed specifically.
    {formatted_metadata}
    """
    
    result = llm.invoke(prompt).content

    url_pattern = r'(https?://[^\s",]+)'
    urls = re.findall(url_pattern, result)

    tldr_articles = [article for article in potential_articles if article['url'] in urls]
    state["tldr_articles"] = tldr_articles

    return state

def summarize_articles_parallel(state: GraphState) -> GraphState:
    """
    Summarize the articles based on their full text.
    (Note: Kept synchronous in this iteration as per original code pattern).
    """
    llm = ChatGroq(model=settings.groq_model, api_key=settings.grok_api_key)
    tldr_articles = state.get("tldr_articles", [])

    prompt = """
    Create a * bulleted summarizing tldr for the article:
    {text}
      
    Be sure to follow the following format exactly with nothing else:
    {title}
    {url}
    * tl;dr bulleted summary
    * use bullet points for each sentence
    """

    for i in range(len(tldr_articles)):
        text = tldr_articles[i]["text"]
        title = tldr_articles[i]["title"]
        url = tldr_articles[i]["url"]
        
        result = llm.invoke(prompt.format(title=title, url=url, text=text))
        tldr_articles[i]["summary"] = result.content

    state["tldr_articles"] = tldr_articles
    return state

def format_results(state: GraphState) -> GraphState:
    """
    Format the final result string for display.
    """
    q_list = [params.get("q", "") for params in state.get("past_searches", [])]
    formatted_results = f"Here are the top {len(state.get('tldr_articles', []))} articles based on search terms:\n{', '.join(q_list)}\n\n"

    tldr_articles = state.get("tldr_articles", [])
    tldr_text = "\n\n".join([article.get('summary', '') for article in tldr_articles])

    formatted_results += tldr_text
    state["formatted_results"] = formatted_results

    return state
