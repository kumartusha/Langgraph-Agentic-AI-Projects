from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore
from src.models.state import State, Router
from src.config.settings import settings

def format_few_shot_examples(examples):
    formatted_examples = []
    for eg in examples:
        email = eg.value['email']
        label = eg.value['label']
        formatted_examples.append(
            f"From: {email['author']}\nSubject: {email['subject']}\nBody: {email['email_thread'][:300]}...\n\nClassification: {label}"
        )
    return "\n\n".join(formatted_examples)

def triage_email_node(state: State, config: dict, store: InMemoryStore) -> dict:
    email = state["email_input"]
    user_id = config["configurable"].get("langgraph_user_id", settings.LANGGRAPH_USER_ID)
    
    # Procedural memory: Retrieve the current triage prompt
    current_prompt_template = store.get(("email_assistant", user_id, "prompts"), "triage_prompt").value
    
    # Episodic memory: Retrieve relevant examples 
    namespace = ("email_assistant", user_id, "examples")
    examples = store.search(namespace, query=str(email))
    formatted_examples = format_few_shot_examples(examples)
    
    # Format the prompt
    prompt = PromptTemplate.from_template(current_prompt_template).format(examples=formatted_examples, **email)
    messages = [HumanMessage(content=prompt)]
    
    # Invoke the LLM with structured output (Router)
    llm = ChatOpenAI(model=settings.MODEL_NAME, temperature=0, openai_api_key=settings.OPENAI_API_KEY)
    llm_router = llm.with_structured_output(Router)
    
    result = llm_router.invoke(messages)
    return {"triage_result": result.classification}
