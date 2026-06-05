from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

from src.config.settings import settings
from src.models.state import State

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    api_key=settings.grok_api_key.get_secret_value()
)

def chunk_large_text(text: str, chunk_size: int = 100000, overlap: int = 1000) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

summarization_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "Summarize the provided article text focusing on the main events, key people involved, "
        "and any important statistics in 150-200 words. Use a neutral tone suitable for a journalistic report:\n\n"
        "Article text:\n{text}\n\n"
    )
)

combine_summarization_prompt = PromptTemplate(
    input_variables=["summaries"],
    template=(
        "Combine the provided summaries into a single coherent summary that captures the main events, key people involved, "
        "and important statistics in 150-200 words. Use a neutral tone suitable for a journalistic report:\n\n"
        "Summaries:\n{summaries}\n\n"
    )
)

summarization_pipeline = summarization_prompt | llm
combine_summarization_pipeline = combine_summarization_prompt | llm

def combine_summaries(summaries: List[str]) -> str:
    if len(summaries) == 1:
        return summaries[0]
    
    summaries_text = ""
    for i, summary in enumerate(summaries):
        summaries_text += f"Summary {i + 1}:\n{summary}\n\n"
    
    full_summary = combine_summarization_pipeline.invoke({"summaries": summaries_text})
    return full_summary.content

def summarize_article(article_text: str, article_chunks: List[str] = None) -> str:
    if not article_chunks:
        article_chunks = chunk_large_text(article_text)

    summaries = []
    for chunk in article_chunks:
        summary = summarization_pipeline.invoke({"text": chunk})
        summaries.append(summary.content)

    return combine_summaries(summaries)

def summarization_node(state: State) -> State:
    article_text = state["article_text"]
    chunks = state.get("chunks", [])
    if not chunks:
        chunks = chunk_large_text(article_text)
    
    summary_result = summarize_article(article_text, chunks)
    return {"summary_result": summary_result, "chunks": chunks}
