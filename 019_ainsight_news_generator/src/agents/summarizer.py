from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.settings import settings
from src.models.state import Article, GraphState

class Summarizer:
    """
    Agent that processes articles and generates accessible summaries
    using Groq models
    """
    
    def __init__(self):
        self.llm = ChatGroq(
            model=settings.groq_model, 
            api_key=settings.grok_api_key,
            temperature=0.1,
            max_tokens=600
        )
        self.system_prompt = """
        You are an AI expert who makes complex topics accessible 
        to general audiences. Summarize this article in 2-3 sentences, focusing on the key points 
        and explaining any technical terms simply.
        """
    
    def summarize(self, article: Article) -> str:
        """
        Generates an accessible summary of a single article
        
        Args:
            article (Article): Article to summarize
            
        Returns:
            str: Generated summary
        """
        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Title: {article.title}\n\nContent: {article.content}")
        ])
        return response.content

def summarize_node(state: GraphState) -> GraphState:
    """
    Node for article summarization
    
    Args:
        state (GraphState): Current workflow state
        
    Returns:
        GraphState: Updated state with summaries
    """
    summarizer = Summarizer()
    state['summaries'] = []
    
    for article in state.get('articles', []):
        summary = summarizer.summarize(article)
        state['summaries'].append({
            'title': article.title,
            'summary': summary,
            'url': article.url
        })
        
    return state
