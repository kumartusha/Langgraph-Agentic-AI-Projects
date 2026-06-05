import sys
import asyncio
from src.graph import app

async def run_workflow(query: str, target_url: str):
    """Run the LangGraph workflow"""
    initial_state = {
        'messages': [],
        'query': query,
        'actions': [],
        'target_url': target_url,
        'current_action': 0,
        'current_action_code': "",
        'aggregated_raw_actions': "",
        'script': "",
        'website_state': "",
        'error_message': "",
        'test_name': "",
        'report': ""
    }

    print(f"🚀 Starting Web Testing Agent Workflow against {target_url}...")
    result = await app.ainvoke(initial_state)
    return result

if __name__ == "__main__":
    if len(sys.argv) > 2:
        query = sys.argv[1]
        target_url = sys.argv[2]
        
        result = asyncio.run(run_workflow(query, target_url))
        print("\n" + "="*50)
        print(result.get("report", "No report generated."))
        print("="*50 + "\n")
    else:
        print("🌐 Welcome to the Web Testing Agent!")
        print("This agent writes and executes End-to-End Playwright tests automatically.")
        
        while True:
            try:
                target_url = input("\nEnter Target URL (e.g. https://example.com) or 'exit': ")
                if target_url.lower() in ['exit', 'quit']:
                    break
                if not target_url.strip():
                    continue
                
                query = input(f"What should I test on {target_url}? (e.g. 'Test if the title exists'): ")
                if query.lower() in ['exit', 'quit']:
                    break
                if not query.strip():
                    continue
                
                result = asyncio.run(run_workflow(query, target_url))
                
                print("\n" + "="*50)
                print(result.get("report", "No report generated."))
                print("="*50 + "\n")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
