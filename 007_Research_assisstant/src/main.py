import argparse
import asyncio
import sys

from src.config import logger
from src.graph import create_workflow

async def run_query(query: str):
    logger.info(f"Starting research assistant for query: '{query}'")
    app = create_workflow()
    
    print(f"\n\033[96m--- Starting Research Task ---\033[0m\n\033[93mQuery:\033[0m {query}\n")
    
    all_messages = []
    # Stream the results
    async for chunk in app.astream({"messages": [query]}, stream_mode="updates"):
        for node_name, updates in chunk.items():
            if messages := updates.get("messages"):
                all_messages.extend(messages)
                for message in messages:
                    # Leverage built-in langchain pretty_print if available
                    if hasattr(message, "pretty_print"):
                        print(f"\033[95m[Node: {node_name}]\033[0m")
                        message.pretty_print()
                    else:
                        print(f"\033[95m[Node: {node_name}]\033[0m \033[92m[{message.type}]\033[0m {message.content}")
                    print("\n")
                    
    print("\n\033[92m--- Workflow Completed ---\033[0m\n")

def main():
    parser = argparse.ArgumentParser(description="Research Assistant CLI")
    parser.add_argument(
        "--query", 
        type=str, 
        required=True, 
        help="The research question or task you want to execute."
    )
    args = parser.parse_args()
    
    try:
        asyncio.run(run_query(args.query))
    except KeyboardInterrupt:
        print("\n\033[91mProcess interrupted by user.\033[0m")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        print(f"\n\033[91mExecution failed:\033[0m {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

## for run this project.
# python3 -m src.main --query "Your research question here"
