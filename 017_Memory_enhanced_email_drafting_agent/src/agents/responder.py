from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from src.tools.email_tools import get_tools
from src.config.settings import settings

def create_agent_prompt(state, config, store):
    messages = state['messages']
    user_id = config["configurable"].get("langgraph_user_id", settings.LANGGRAPH_USER_ID)
    
    # Get the current response prompt from procedural memory
    system_prompt = store.get(("email_assistant", user_id, "prompts"), "response_prompt").value
    
    return [{"role": "system", "content": system_prompt}] + messages

def create_responder_agent(store):
    llm = ChatOpenAI(model=settings.MODEL_NAME, temperature=0, openai_api_key=settings.OPENAI_API_KEY)
    
    # Create the React Agent passing the procedural prompt via create_agent_prompt
    response_agent = create_react_agent(
        tools=get_tools(),
        prompt=create_agent_prompt,
        store=store,
        model=llm
    )
    return response_agent
