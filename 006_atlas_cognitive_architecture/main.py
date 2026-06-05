"""
main.py
-------
Entry point for the ATLAS Academic Task and Learning Agent System.

Usage:
    python main.py

    Make sure you have a valid .env file with GROK_API_KEY set,
    and the three data files present in the data/ directory:
        data/profile.json
        data/calendar_events.json
        data/task.json
"""

import asyncio
from config.settings import configure_api_keys
from runner.run_system import load_json_and_test


def main():
    # 1. Validate API keys before doing anything else
    if not configure_api_keys():
        print("\n❌ Cannot start ATLAS — API key missing.")
        print("   Add GROK_API_KEY=<your_key> to a .env file in the atlas/ directory.")
        return

    # 2. Run the full multi-agent workflow
    asyncio.run(load_json_and_test())


if __name__ == "__main__":
    main()
