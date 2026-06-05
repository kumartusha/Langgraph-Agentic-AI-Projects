import os
from typing import List, Dict
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.settings import settings
from src.models.state import GraphState, Summary

class Publisher:
    """
    Agent that compiles summaries into a formatted report 
    and saves it to disk
    """
    def __init__(self):
        self.llm = ChatGroq(
            model=settings.groq_model, 
            api_key=settings.grok_api_key,
            temperature=0.1,
            max_tokens=800
        )
        
    def create_report(self, summaries: List[Summary]) -> str:
        """
        Creates and saves a formatted markdown report
        
        Args:
            summaries (List[Summary]): Collection of article summaries
            
        Returns:
            str: Generated report content
        """
        prompt = """
        Create a weekly AI/ML news report for the general public. 
        Format it with:
        1. A brief introduction
        2. The main news items with their summaries
        3. Links for further reading
        
        Make it engaging and accessible to non-technical readers.
        """
        
        # Format summaries for the LLM
        summaries_text = "\n\n".join([
            f"Title: {item['title']}\nSummary: {item['summary']}\nSource: {item['url']}"
            for item in summaries
        ])
        
        # Generate report
        response = self.llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=summaries_text)
        ])
        
        # Add metadata and save
        current_date = datetime.now().strftime("%Y-%m-%d")
        markdown_content = f"""
        Generated on: {current_date}

        {response.content}
        """
        
        # Ensure we save to output directory if needed, for now just in cwd
        filename = f"ai_news_report_{current_date}.md"
        with open(filename, 'w') as f:
            f.write(markdown_content)
        
        return response.content

def publish_node(state: GraphState) -> GraphState:
    """
    Node for report generation
    
    Args:
        state (GraphState): Current workflow state
        
    Returns:
        GraphState: Updated state with final report
    """
    publisher = Publisher()
    report_content = publisher.create_report(state.get('summaries', []))
    state['report'] = report_content
    return state
