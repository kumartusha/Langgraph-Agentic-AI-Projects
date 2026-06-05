from langmem import create_multi_prompt_optimizer
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore
from src.config.settings import settings
from rich.console import Console

console = Console()

def optimize_prompts(feedback: str, config: dict, store: InMemoryStore):
    """Improve our prompts based on feedback using langmem prompt optimizer."""
    user_id = config["configurable"].get("langgraph_user_id", settings.LANGGRAPH_USER_ID)
    
    # Get current prompts
    triage_prompt = store.get(("email_assistant", user_id, "prompts"), "triage_prompt").value
    response_prompt = store.get(("email_assistant", user_id, "prompts"), "response_prompt").value
    
    # Sample relevant to the issue
    sample_email = {
        "author": "Alice Smith <alice.smith@company.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Quick question about API documentation",
        "email_thread": "Hi John, I was reviewing the API documentation and noticed a few endpoints are missing. Could you help? Thanks, Alice",
    }
    
    llm = ChatOpenAI(model=settings.MODEL_NAME, temperature=0, openai_api_key=settings.OPENAI_API_KEY)
    optimizer = create_multi_prompt_optimizer(llm)
    
    conversation = [
        {"role": "system", "content": response_prompt},
        {"role": "user", "content": f"I received this email: {sample_email}"},
        {"role": "assistant", "content": "How can I assist you today?"}
    ]
    
    prompts = [
        {"name": "triage", "prompt": triage_prompt},
        {"name": "response", "prompt": response_prompt}
    ]
    
    try:
        trajectories = [(conversation, {"feedback": feedback})]
        result = optimizer.invoke({"trajectories": trajectories, "prompts": prompts})
        
        improved_triage_prompt = next(p["prompt"] for p in result if p["name"] == "triage")
        improved_response_prompt = next(p["prompt"] for p in result if p["name"] == "response")
        console.print("[bold green]Successfully optimized prompts via LangMem Optimizer![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]API error during optimization: {e}[/bold red]")
        console.print("[yellow]Using manual prompt improvement as fallback...[/yellow]")
        
        improved_triage_prompt = triage_prompt + "\n\nNote: Emails about API documentation or missing endpoints are high priority and should ALWAYS be classified as 'respond'."
        improved_response_prompt = response_prompt + "\n\nWhen responding to emails about documentation or API issues, acknowledge the specific issue mentioned and offer specific assistance rather than generic responses."
    
    # Store the improved prompts back to memory
    store.put(("email_assistant", user_id, "prompts"), "triage_prompt", improved_triage_prompt)
    store.put(("email_assistant", user_id, "prompts"), "response_prompt", improved_response_prompt)
    
    console.print("\n[bold cyan]Improved Triage Prompt:[/bold cyan]")
    console.print(f"{improved_triage_prompt[:150]}...\n")
    console.print("[bold cyan]Improved Response Prompt:[/bold cyan]")
    console.print(f"{improved_response_prompt[:150]}...\n")
    
    return "Prompts improved based on feedback!"
