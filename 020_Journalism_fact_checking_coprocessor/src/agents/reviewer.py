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

# --- Quote Extraction ---
quote_extraction_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "Identify direct quotes in the following content, noting the speaker's name "
        "and the context of each quote. If there are no quotes, return 'No quotes found'.\n\n"
        "Text: {text}"
    )
)

quote_extraction_pipeline = quote_extraction_prompt | llm

def quote_extraction_article(article_text: str, chunks: List[str] = None) -> List[str]:
    if not chunks:
        chunks = chunk_large_text(article_text)
    
    quote_results = []
    for chunk in chunks:
        quote_result = quote_extraction_pipeline.invoke({"text": chunk})
        quote_results.append(quote_result.content)
    
    return quote_results

def quote_extraction_node(state: State) -> State:
    article_text = state["article_text"]
    chunks = state.get("chunks", [])
    if not chunks:
        chunks = chunk_large_text(article_text)
        
    quote_extraction_results = quote_extraction_article(article_text, chunks)
    
    return {"quote_extraction_result": quote_extraction_results, "chunks": chunks}

# --- Grammar and Bias Review ---
review_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "Review the following article for grammar, spelling, punctuation, and bias. "
        "Provide feedback on each aspect in form of a list of the issues found and some suggestions for improvement.\n\n"
        "{text}"
    )
)

grammar_and_bias_review = review_prompt | llm

def grammar_and_bias_analysis_article(article_text: str, chunks: List[str] = None) -> List[str]:
    if not chunks:
        chunks = chunk_large_text(article_text)
    
    review_results = []
    for chunk in chunks:
        review_result = grammar_and_bias_review.invoke({"text": chunk})
        review_results.append(review_result.content)
    
    return review_results

def grammar_and_bias_review_node(state: State) -> State:
    article_text = state["article_text"]
    chunks = state.get("chunks", [])
    if not chunks:
        chunks = chunk_large_text(article_text)
        
    grammar_and_bias_review_results = grammar_and_bias_analysis_article(article_text, chunks)
    
    return {"grammar_and_bias_review_result": grammar_and_bias_review_results, "chunks": chunks}
