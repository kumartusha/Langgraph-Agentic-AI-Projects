from langgraph.store.memory import InMemoryStore
from src.config.settings import settings

def init_memory_store() -> InMemoryStore:
    # Initialize the memory store with the specified embedding model
    store = InMemoryStore(index={"embed": f"openai:{settings.EMBEDDING_MODEL}"})
    return store

def populate_initial_memory(store: InMemoryStore, user_id: str):
    # Procedural Memory: Store baseline prompts
    initial_triage_prompt = """You are an email triage assistant. Classify the following email:
From: {author}
To: {to}
Subject: {subject}
Body: {email_thread}

Classify as 'ignore', 'notify', or 'respond'.

Here are some examples of previous classifications:
{examples}
"""
    initial_response_prompt = """You are a helpful assistant. Use the tools available, including memory tools, to assist the user."""
    
    store.put(("email_assistant", user_id, "prompts"), "triage_prompt", initial_triage_prompt)
    store.put(("email_assistant", user_id, "prompts"), "response_prompt", initial_response_prompt)

    # Episodic Memory: Store a baseline example
    spam_example = {
        "email": {
            "author": "Spammy Marketer <spam@example.com>",
            "to": "John Doe <john.doe@company.com>",
            "subject": "BIG SALE!!!",
            "email_thread": "Buy our product now and get 50% off!",
        },
        "label": "ignore",
    }
    store.put(("email_assistant", user_id, "examples"), "spam_example", spam_example)
