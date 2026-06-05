import asyncio
import sys
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

from src.workflow.graph import create_workflow

console = Console()

async def run_news_agent(query: str, num_searches_remaining: int = 10, num_articles_tldr: int = 3):
    """
    Run the LangGraph workflow and display results using rich UI.
    """
    console.print(Panel(f"[bold cyan]Starting News Summarization Agent[/bold cyan]\nQuery: [yellow]{query}[/yellow]", border_style="cyan"))

    initial_state = {
        "news_query": query,
        "num_searches_remaining": num_searches_remaining,
        "newsapi_params": {},
        "past_searches": [],
        "articles_metadata": [],
        "scraped_urls": [],
        "num_articles_tldr": num_articles_tldr,
        "potential_articles": [],
        "tldr_articles": [],
        "formatted_results": "No articles with text found."
    }

    app = create_workflow()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("[cyan]Initializing LangGraph workflow...", total=None)

        try:
            # We iterate over the astream to update the UI based on the active node
            async for output in app.astream(initial_state):
                for node_name, state in output.items():
                    if node_name == "generate_newsapi_params":
                        progress.update(task, description=f"[cyan]Generating API parameters for search: [yellow]'{state.get('newsapi_params', {}).get('q', '')}'[/yellow]")
                    elif node_name == "retrieve_articles_metadata":
                        progress.update(task, description=f"[magenta]Fetching metadata from NewsAPI... found [bold]{len(state.get('articles_metadata', []))}[/bold] new articles")
                    elif node_name == "retrieve_articles_text":
                        progress.update(task, description=f"[blue]Scraping full text via BeautifulSoup... Total valid articles: [bold]{len(state.get('potential_articles', []))}[/bold]")
                    elif node_name == "select_top_urls":
                        progress.update(task, description="[green]LLM evaluating relevance and selecting top articles...")
                    elif node_name == "summarize_articles_parallel":
                        progress.update(task, description="[green]LLM generating bulleted TL;DR summaries...")
                    elif node_name == "format_results":
                        progress.update(task, description="[cyan]Formatting final output...")
            
            # Retrieve final state from the last output
            final_state = state
            
        except Exception as e:
            console.print(f"[bold red]An error occurred during graph execution: {str(e)}[/bold red]")
            return

    console.print(Panel("[bold green]Workflow Completed Successfully![/bold green]", border_style="green"))
    console.print(Markdown(final_state.get("formatted_results", "No results found.")))


if __name__ == "__main__":
    from src.config.settings import settings
    if not settings.grok_api_key or not settings.newsapi_key:
        console.print("[bold red]ERROR: Missing GROK_API_KEY or NEWSAPI_KEY.[/bold red]")
        console.print("[yellow]Please add `NEWSAPI_KEY=your_key` and `GROK_API_KEY=your_key` to your `/Users/apple/Desktop/DesktopBackup/GenerativeAI_Foundation/.env` file![/yellow]")
        sys.exit(1)

    default_query = "What are the top AI and Generative AI news of the week?"
    query = sys.argv[1] if len(sys.argv) > 1 else default_query
    
    asyncio.run(run_news_agent(query, num_articles_tldr=3))


# How to run the agent

# cd /Users/apple/Desktop/DesktopBackup/GenerativeAI_Foundation/Langgraph/projects/018_news_summarization
# python3 main.py "Latest updates on SpaceX and Starship"
# python3 main.py "Latest AI news"
