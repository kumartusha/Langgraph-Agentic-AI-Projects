import time
from typing import List, Dict, Any
from functools import lru_cache
from duckduckgo_search import DDGS
from pydantic import BaseModel, Field

from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_groq import ChatGroq

from src.config.settings import settings
from src.models.state import State, FactCheckStatement
from src.agents.summarizer import summarize_article, chunk_large_text

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    api_key=settings.grok_api_key.get_secret_value()
)

ddgs = DDGS()

class FactCheckStatementModel(BaseModel):
    statement: str = Field(description="Original statement")
    status: str = Field(description="confirmed | refuted | unverifiable | vague")
    explanation: str = Field(description="Brief explanation of findings or reason for vagueness")
    suggested_keywords: List[str] = Field(description="Keywords for further search")

class FactCheckResultModel(BaseModel):
    result: List[FactCheckStatementModel]

@lru_cache
def search_ddg(keywords: str, max_results: int = 1) -> List[Dict[str, Any]]:
    try:
        text_results = ddgs.text(keywords=keywords, max_results=max_results)
    except Exception as e:
        time.sleep(5)
        try:
            text_results = ddgs.text(keywords=keywords, max_results=max_results)
        except Exception:
            return [{}]
    
    return text_results if text_results else [{}]

def search_and_summarize(keywords: str, max_results: int = 1) -> List[Dict[str, Any]]:
    text_results = search_ddg(keywords, max_results)
    
    results = []
    for result in text_results:
        if not result or 'href' not in result:
            continue
        try:
            loader = WebBaseLoader([str(result['href'])])
            docs = loader.load()
            if not docs:
                continue
            html_content = docs[0].page_content
            
            # Simple summarization instead of BeautifulSoupTransformer to reduce complexity 
            # and dependency issues if we don't need strict extraction
            summary_result = summarize_article(html_content)
            
            results.append({
                "title": result.get('title', ''),
                "url": result.get('href', ''),
                "summary": summary_result
            })
        except Exception:
            pass

    return results

fact_checking_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "Fact-check the texts provided. For each statement, identify any factual inaccuracies, misleading information, "
        "unsupported claims, or vague language lacking specific details. Confirm accuracy for each claim where possible, "
        "or provide suggestions for further searches. Flag statements as 'vague' if they are overly broad or lacking "
        "critical specifics (e.g., missing names, dates, or descriptions of technologies)."
        "Suggest keyword if you can't confirm or refute the statement.\n\n"
        "{text}\n\n"
    )
)

structured_output_llm = llm.with_structured_output(FactCheckResultModel)
fact_checking_pipeline = fact_checking_prompt | structured_output_llm

def fact_check_article(article_text: str, chunks: List[str] = None) -> List[FactCheckStatement]:
    if not chunks:
        chunks = chunk_large_text(article_text)
    
    fact_check_results = []
    for chunk in chunks:
        response = fact_checking_pipeline.invoke({"text": chunk})
        if not response or not hasattr(response, 'result'):
            continue
            
        for statement_model in response.result:
            statement = statement_model.dict()
            suggested_keywords = statement.get('suggested_keywords', [])
            
            if suggested_keywords:
                statement['search_results'] = [
                    search_and_summarize(keyword) for keyword in suggested_keywords
                ]
            
            fact_check_results.append(statement)

    return fact_check_results

def fact_checking_node(state: State) -> State:
    article_text = state["article_text"]
    chunks = state.get("chunks", [])
    if not chunks:
        chunks = chunk_large_text(article_text)
        
    fact_checking_results = fact_check_article(article_text, chunks)
    
    return {"fact_check_result": fact_checking_results, "chunks": chunks}
