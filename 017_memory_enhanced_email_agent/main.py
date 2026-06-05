import os
from rich.console import Console
from rich.panel import Panel
from src.config.settings import settings
from src.memory.store import init_memory_store, populate_initial_memory
from src.workflow.graph import create_email_workflow
from src.agents.optimizer import optimize_prompts

console = Console()

def run_simulation():
    console.print(Panel.fit("[bold blue]Memory Enhanced Email Agent (017)[/bold blue]", border_style="blue"))

    # 1. Initialize Store
    console.print("\n[bold yellow]1. Initializing Memory Store (Episodic & Procedural)...[/bold yellow]")
    store = init_memory_store()
    populate_initial_memory(store, settings.LANGGRAPH_USER_ID)
    
    # Sample Email to test
    email_input = {
        "author": "Alice Smith <alice.smith@company.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Quick question about API documentation",
        "email_thread": "Hi John,\n\nI was reviewing the API documentation and noticed a few endpoints are missing. Could you help?\n\nThanks,\nAlice",
    }
    config = {"configurable": {"langgraph_user_id": settings.LANGGRAPH_USER_ID}}
    inputs = {"email_input": email_input, "messages": []}

    # 2. Run Before Optimization
    console.print("\n[bold yellow]2. Running Original Agent (Before Optimization)...[/bold yellow]")
    agent_v1 = create_email_workflow(store)
    
    for output in agent_v1.stream(inputs, config=config):
        for key, value in output.items():
            console.print(f"[bold magenta]--- {key.upper()} ---[/bold magenta]")
            console.print(value)

    # 3. Add Edge-case to Episodic Memory
    console.print("\n[bold yellow]3. Adding Edge Case to Episodic Memory...[/bold yellow]")
    api_doc_example = {
        "email": {
            "author": "Developer <dev@company.com>",
            "to": "John Doe <john.doe@company.com>",
            "subject": "API Documentation Issue", 
            "email_thread": "Found missing endpoints in the API docs. Need urgent update.",
        },
        "label": "respond",
    }
    store.put(("email_assistant", settings.LANGGRAPH_USER_ID, "examples"), "api_doc_example", api_doc_example)
    console.print("✅ Added 'api_doc_example' to memory")

    # 4. Optimize Prompts (Procedural Memory)
    console.print("\n[bold yellow]4. Optimizing Prompts via Human Feedback...[/bold yellow]")
    feedback = """The agent didn't properly recognize that emails about API documentation issues 
    are high priority and require immediate attention. When an email mentions 
    'API documentation', it should always be classified as 'respond' with a helpful tone.
    Also, instead of just responding with 'How can I assist you today?', the agent should 
    acknowledge the specific documentation issue mentioned and offer assistance."""
    
    optimize_prompts(feedback, config, store)

    # 5. Run After Optimization
    console.print("\n[bold yellow]5. Running Agent v2 (After Optimization)...[/bold yellow]")
    agent_v2 = create_email_workflow(store)
    
    for output in agent_v2.stream(inputs, config=config):
        for key, value in output.items():
            console.print(f"[bold magenta]--- {key.upper()} ---[/bold magenta]")
            console.print(value)
            
    console.print("\n[bold green]Simulation Complete![/bold green]")

if __name__ == "__main__":
    run_simulation()
