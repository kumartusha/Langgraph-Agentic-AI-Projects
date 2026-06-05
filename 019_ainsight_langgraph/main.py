import sys
from rich.console import Console
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live

console = Console()

def main():
    console.print(Panel.fit("[bold cyan]019 AInsight LangGraph[/bold cyan]\n[italic]Autonomous AI/ML News Researcher[/italic]", border_style="cyan"))

    # Validate environments
    try:
        from src.config.settings import settings
        if not settings.grok_api_key or not settings.tavily_api_key:
            console.print("[bold red]ERROR: Missing GROK_API_KEY or TAVILY_API_KEY.[/bold red]")
            console.print("[yellow]Please add `GROK_API_KEY=your_key` and `TAVILY_API_KEY=your_key` to your `.env` file![/yellow]")
            sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {str(e)}")
        sys.exit(1)

    from src.workflow.graph import create_workflow

    with Live(Spinner("dots", text="[yellow]Initializing LangGraph Workflow...[/yellow]"), console=console, refresh_per_second=10) as live:
        workflow = create_workflow()
        
        live.update(Spinner("dots", text="[cyan]Searching Tavily for latest AI/ML news...[/cyan]"))
        # Execute workflow
        final_state = workflow.invoke({
            "articles": None,
            "summaries": None,
            "report": None
        })
        
        live.update(Spinner("dots", text="[green]Report generated successfully![/green]"))

    # Display results
    console.print("\n[bold magenta]=== AI/ML Weekly News Report ===[/bold magenta]\n")
    console.print(final_state.get('report', 'No report generated.'))

if __name__ == "__main__":
    main()



# How to Run

# cd /Users/apple/Desktop/DesktopBackup/GenerativeAI_Foundation/Langgraph/projects/019_ainsight_langgraph
# python3 main.py