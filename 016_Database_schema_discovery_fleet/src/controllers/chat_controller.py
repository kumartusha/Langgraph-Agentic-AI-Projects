from fastapi import APIRouter
from src.models.domain_models import ChatRequest, ChatResponse
from src.services.supervisor_agent import SupervisorAgent

router = APIRouter()
supervisor = SupervisorAgent()
app_graph = supervisor.compile_graph()

# We maintain state in-memory across requests for simplicity (global dict)
# In production, use a persistent checkpointer (e.g., Redis or Sqlite checkpointer)
conversation_state_store = {}

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global conversation_state_store
    
    # Initialize state if not present
    if not conversation_state_store:
        conversation_state_store = {
            "question": request.question,
            "input_type": "",
            "plan": [],
            "db_results": "",
            "response": "",
            "db_graph": None
        }
    else:
        # Update question for the next step
        conversation_state_store["question"] = request.question

    # Invoke graph with current state
    new_state = app_graph.invoke(conversation_state_store)
    
    # Update our global state
    conversation_state_store = new_state
    
    return ChatResponse(
        response=new_state.get("response", "No response generated."),
        db_results=new_state.get("db_results"),
        plan=new_state.get("plan")
    )
