from typing import TypedDict, List, Optional, Any

class FactCheckStatement(TypedDict):
    statement: str
    status: str
    explanation: str
    suggested_keywords: List[str]
    search_results: Optional[List[Any]]

class SystemAction(TypedDict):
    actions: List[str]

class State(TypedDict):
    current_query: str
    article_text: str
    chunks: List[str]
    actions: List[str]
    summary_result: Optional[str]
    fact_check_result: Optional[List[FactCheckStatement]]
    tone_analysis_result: Optional[List[str]]
    quote_extraction_result: Optional[List[str]]
    grammar_and_bias_review_result: Optional[List[str]]
    review_result: Optional[str]
