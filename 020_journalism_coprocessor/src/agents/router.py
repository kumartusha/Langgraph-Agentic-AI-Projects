from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from src.config.settings import settings
from src.models.state import State, SystemAction

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    api_key=settings.grok_api_key.get_secret_value()
)

action_prompt = PromptTemplate(
    input_variables=["input_text"],
    template=(
        "Identify the user's intended actions based on their input and return the actions in the following JSON format:\n"
        "{{\n"
        '  "actions": ["<summarization | fact-checking | tone-analysis | quote-extraction | grammar-and-bias-review | no-action-required | invalid>"]\n'
        "}}\n\n"
        "Guidelines:\n"
        "- If the user requests all actions or says 'everything' or 'full report,' respond with the list of all individual actions:\n"
        '{{\n'
        '    "actions": ["summarization", "fact-checking", "tone-analysis", "quote-extraction", "grammar-and-bias-review"]\n'
        "}}\n"
        "- If the user input requests multiple specific actions, list each action requested (e.g., 'summarization' and 'tone analysis' together as ['summarization', 'tone-analysis']).\n"
        "- If the user’s input does not relate to any accessible action, respond with:\n"
        '{{\n'
        '    "actions": ["invalid"]\n'
        "}}\n"
        "- If the user's input does not require any specific action, or wants to end the conversation, respond with:\n"
        '{{\n'
        '    "actions": ["no-action-required"]\n'
        "}}\n\n"
        "Important:\n"
        "- Only list all actions ('summarization', 'fact-checking', 'tone-analysis', 'quote-extraction', 'grammar-and-bias-review') if the user explicitly requests a comprehensive overview or all actions.\n"
        "- List only the actions explicitly requested by the user without inferring additional ones.\n\n"
        "Input text:\n{input_text}"
    )
)

action_pipeline = action_prompt | llm.with_structured_output(SystemAction)

def get_user_actions(input_text: str) -> List[str]:
    system_actions = action_pipeline.invoke({"input_text": input_text})
    if not system_actions or not hasattr(system_actions, 'actions'):
        # Handle dict or obj
        if isinstance(system_actions, dict):
            return system_actions.get("actions", ["invalid"])
        return getattr(system_actions, "actions", ["invalid"])
    return system_actions.actions

def categorize_user_input(state: State) -> State:
    query = state["current_query"]
    actions = get_user_actions(query)
    return {"actions": actions}
