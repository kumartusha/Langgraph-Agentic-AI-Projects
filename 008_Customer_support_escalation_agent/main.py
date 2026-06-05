import os
import sys
from typing import Dict, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.constants import START

# Load environment variables (such as GROK_API_KEY)
load_dotenv()
if "GROK_API_KEY" in os.environ:
    os.environ["GROK_API_KEY"] = os.environ.get("GROK_API_KEY")

# ==========================================
# 1. State Definition
# ==========================================
class State(TypedDict):
    query: str
    category: str
    sentiment: str
    response: str

# ==========================================
# 2. Node Functions
# ==========================================
def categorize(state: State) -> State:
    """Categorize the customer query into Technical, Billing, or General."""
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following customer query into one of these categories: "
        "Technical, Billing, General. Query: {query}"
    )
    chain = prompt | ChatGroq(temperature=0, model="qwen/qwen3-32b", api_key=os.getenv("GROK_API_KEY"))
    category = chain.invoke({"query": state["query"]}).content
    return {"category": category}

def analyze_sentiment(state: State) -> State:
    """Analyze the sentiment of the customer query as Positive, Neutral, or Negative."""
    prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of the following customer query. "
        "Respond with either 'Positive', 'Neutral', or 'Negative'. Query: {query}"
    )
    chain = prompt | ChatGroq(temperature=0, model="qwen/qwen3-32b",  api_key=os.getenv("GROK_API_KEY"))
    sentiment = chain.invoke({"query": state["query"]}).content
    return {"sentiment": sentiment}

def handle_technical(state: State) -> State:
    """Provide a technical support response to the query."""
    prompt = ChatPromptTemplate.from_template(
        "Provide a technical support response to the following query: {query}"
    )
    chain = prompt | ChatGroq(temperature=0, model="qwen/qwen3-32b",  api_key=os.getenv("GROK_API_KEY"))
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}

def handle_billing(state: State) -> State:
    """Provide a billing support response to the query."""
    prompt = ChatPromptTemplate.from_template(
        "Provide a billing support response to the following query: {query}"
    )
    chain = prompt | ChatGroq(temperature=0, model="qwen/qwen3-32b",  api_key=os.getenv("GROK_API_KEY"))
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}

def handle_general(state: State) -> State:
    """Provide a general support response to the query."""
    prompt = ChatPromptTemplate.from_template(
        "Provide a general support response to the following query: {query}"
    )
    chain = prompt | ChatGroq(temperature=0, model="qwen/qwen3-32b",  api_key=os.getenv("GROK_API_KEY"))
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}

def escalate(state: State) -> State:
    """Escalate the query to a human agent due to negative sentiment."""
    return {"response": "This query has been escalated to a human agent due to its negative sentiment."}

# ==========================================
# 3. Routing Logic
# ==========================================
def route_query(state: State) -> str:
    """Route the query based on its sentiment and category."""
    if state["sentiment"] == "Negative":
        return "escalate"
    elif state["category"] == "Technical":
        return "handle_technical"
    elif state["category"] == "Billing":
        return "handle_billing"
    else:
        return "handle_general"

# ==========================================
# 4. Graph Setup
# ==========================================
workflow = StateGraph(State)

# Add nodes
workflow.add_node("categorize", categorize)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("handle_technical", handle_technical)
workflow.add_node("handle_billing", handle_billing)
workflow.add_node("handle_general", handle_general)
workflow.add_node("escalate", escalate)

# Add edges
workflow.add_edge(START, "categorize")
workflow.add_edge("categorize", "analyze_sentiment")
workflow.add_conditional_edges(
    "analyze_sentiment",
    route_query,
    {
        "handle_technical": "handle_technical",
        "handle_billing": "handle_billing",
        "handle_general": "handle_general",
        "escalate": "escalate"
    }
)
workflow.add_edge("handle_technical", END)
workflow.add_edge("handle_billing", END)
workflow.add_edge("handle_general", END)
workflow.add_edge("escalate", END)

# Compile the graph
app = workflow.compile()

# ==========================================
# 5. Execution Wrapper
# ==========================================
def run_customer_support(query: str) -> Dict[str, str]:
    """Process a customer query through the LangGraph workflow."""
    results = app.invoke({"query": query})
    return {
        "category": results.get("category", "Unknown"),
        "sentiment": results.get("sentiment", "Unknown"),
        "response": results.get("response", "No response generated.")
    }

# ==========================================
# 6. Command-line Interface
# ==========================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run a single query passed as an argument
        user_query = " ".join(sys.argv[1:])
        print(f"Query: {user_query}")
        result = run_customer_support(user_query)
        print("\n--- Output ---")
        print(f"Category:  {result['category']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Response:  {result['response']}\n")
    else:
        # Interactive mode
        print("🤖 Welcome to the Customer Support Agent (Monolithic)!")
        print("Type 'exit' or 'quit' to stop.\n")
        while True:
            try:
                user_query = input("Enter your query: ")
                if user_query.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                if not user_query.strip():
                    continue
                    
                print("\nProcessing...")
                result = run_customer_support(user_query)
                print("\n--- Output ---")
                print(f"Category:  {result['category']}")
                print(f"Sentiment: {result['sentiment']}")
                print(f"Response:  {result['response']}\n")
                print("-" * 50 + "\n")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
