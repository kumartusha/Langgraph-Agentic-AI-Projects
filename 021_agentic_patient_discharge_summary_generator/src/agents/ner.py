import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import AgentState
from src.models.schemas import ExtractedEntities

# Explicitly load the .env from the GenerativeAI_Foundation folder as requested
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.env"))
load_dotenv(dotenv_path)

# Initialize the Groq LLM with the versatile Llama 3.3 70b model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# Bind the LLM to our Pydantic schema to force structured extraction
structured_llm = llm.with_structured_output(ExtractedEntities)

def medical_ner_node(state: AgentState) -> dict:
    """
    LangGraph node to extract medical entities (diagnoses, medications) from the raw note.
    """
    print("---RUNNING MEDICAL NER AGENT---")
    raw_note = state.get("raw_clinical_note", "")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Medical NLP extraction agent. Your job is to extract medical diagnoses, medications, and procedures from unstructured clinical notes. Be extremely precise. Extract the exact medication names and dosages. If a medication is mentioned as an admission medication OR a discharge medication, extract it."),
        ("human", "Clinical Note:\n{clinical_note}\n\nExtract the medical entities.")
    ])
    
    chain = prompt | structured_llm
    
    # Run the extraction
    result: ExtractedEntities = chain.invoke({"clinical_note": raw_note})
    
    # Convert Pydantic objects to dicts for the LangGraph state
    return {
        "extracted_diagnoses": [diag.model_dump() for diag in result.diagnoses] if result.diagnoses else [],
        "extracted_medications": [med.model_dump() for med in result.medications] if result.medications else [],
        "extracted_procedures": result.procedures if result.procedures else []
    }
