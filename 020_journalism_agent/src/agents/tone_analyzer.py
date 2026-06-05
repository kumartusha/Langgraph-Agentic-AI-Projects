from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from src.config.settings import settings
from src.models.state import State
from src.agents.summarizer import chunk_large_text

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    api_key=settings.grok_api_key.get_secret_value()
)

tone_analysis_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "Analyze the tones of the following article. Does it appear neutral, positive, critical, or opinionated? "
        "Provide a short explanation for each detected tone. "
        "Use specific examples from the article to support your analysis.\n\n{text}"
    )
)

tone_pipeline = tone_analysis_prompt | llm

def tone_analysis_article(article_text: str, chunks: List[str] = None) -> List[str]:
    if not chunks:
        chunks = chunk_large_text(article_text)
    
    tone_results = []
    for chunk in chunks:
        tone_result = tone_pipeline.invoke({"text": chunk})
        tone_results.append(tone_result.content)
    
    return tone_results

def tone_analysis_node(state: State) -> State:
    article_text = state["article_text"]
    chunks = state.get("chunks", [])
    if not chunks:
        chunks = chunk_large_text(article_text)
        
    tone_analysis_results = tone_analysis_article(article_text, chunks)
    
    return {"tone_analysis_result": tone_analysis_results, "chunks": chunks}
