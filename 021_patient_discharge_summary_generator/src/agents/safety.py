import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import AgentState
from pydantic import BaseModel, Field

dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.env"))
load_dotenv(dotenv_path)

class SafetyResult(BaseModel):
    is_safe: bool = Field(description="True if the summary is medically safe and appropriate, False otherwise.")
    feedback: str = Field(description="If unsafe, provide the reason. If safe, output 'Approved'.")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
structured_llm = llm.with_structured_output(SafetyResult)

def safety_node(state: AgentState) -> dict:
    """
    LangGraph node acting as a safety guardrail proxy.
    """
    print("---RUNNING SAFETY AGENT---")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict medical safety reviewer (acting as a LlamaGuard proxy). Review the provided drafted discharge summary. Ensure it does not contain harmful medical advice, dangerous hallucinations, or obvious medication dosage errors. Return is_safe=True if it is a standard, safe discharge summary. Return False if it contains something dangerous."),
        ("human", "Draft Summary:\n{summary}")
    ])
    
    chain = prompt | structured_llm
    result = chain.invoke({"summary": state.get("draft_summary", "")})
    
    return {
        "safety_approved": result.is_safe,
        "safety_feedback": result.feedback
    }
