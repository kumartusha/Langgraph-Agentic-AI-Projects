import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.state import AgentState

dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.env"))
load_dotenv(dotenv_path)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

def drafting_node(state: AgentState) -> dict:
    """
    LangGraph node to draft the final SOAP note based on extracted structured data.
    """
    print("---RUNNING SUMMARY DRAFTING AGENT---")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert physician drafting a patient discharge summary. Using the provided extracted entities, draft a highly professional, well-formatted SOAP (Subjective, Objective, Assessment, Plan) note. Include a specific section called 'Medication Reconciliation Flags' at the end of the note if any exist."),
        ("human", "Diagnoses: {diagnoses}\n\nMedications: {medications}\n\nProcedures: {procedures}\n\nReconciliation Flags: {flags}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    summary = chain.invoke({
        "diagnoses": state.get("extracted_diagnoses", []),
        "medications": state.get("extracted_medications", []),
        "procedures": state.get("extracted_procedures", []),
        "flags": state.get("reconciliation_flags", [])
    })
    
    return {"draft_summary": summary}
