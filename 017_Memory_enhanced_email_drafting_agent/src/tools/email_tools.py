from langchain_core.tools import tool
from langmem import create_manage_memory_tool, create_search_memory_tool
from src.config.settings import settings
from rich.console import Console

console = Console()

@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    console.print(f"[bold green]Sending email to {to}[/bold green] with subject '{subject}'")
    console.print(f"[italic]Content:\n{content}[/italic]\n")
    return f"Email sent to {to} with subject '{subject}'"

@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"

# Create LangMem memory tools (using the configured user ID)
manage_memory_tool = create_manage_memory_tool(namespace=("email_assistant", "{langgraph_user_id}", "collection"))
search_memory_tool = create_search_memory_tool(namespace=("email_assistant", "{langgraph_user_id}", "collection"))

def get_tools():
    return [write_email, check_calendar_availability, manage_memory_tool, search_memory_tool]
