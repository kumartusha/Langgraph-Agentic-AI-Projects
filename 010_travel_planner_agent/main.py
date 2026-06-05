import os
import sys
import json
from typing import Dict, TypedDict, Annotated, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.constants import START

# Load environment variables
load_dotenv()
if "GROK_API_KEY" in os.environ:
    os.environ["GROK_API_KEY"] = os.environ.get("GROK_API_KEY")

# Initialize the ChatGroq model
llm = ChatGroq(temperature=0, model="qwen/qwen3-32b", api_key=os.getenv("GROK_API_KEY"))

# ==========================================
# 1. State Definition
# ==========================================
class PlannerState(TypedDict):
    """Maintains the conversational state and extracted variables for the trip."""
    messages: List[BaseMessage]
    city: str
    interests: List[str]
    itinerary: str

# ==========================================
# 2. Node Functions
# ==========================================
def extract_preferences(state: PlannerState) -> PlannerState:
    """Extract city and interests from the user's message."""
    user_message = state["messages"][-1].content
    prompt = ChatPromptTemplate.from_template(
        "Analyze the following travel request and extract the destination city and the user's interests. "
        "Return the output STRICTLY as a JSON object with two keys: 'city' (string, empty if not found) "
        "and 'interests' (list of strings). Do not include any other text or markdown formatting.\n\n"
        "Request: {request}"
    )
    
    try:
        response = llm.invoke(prompt.format(request=user_message)).content
        # Clean response to parse JSON
        cleaned_response = response.strip().strip("```json").strip("```")
        data = json.loads(cleaned_response)
        
        state["city"] = data.get("city", "")
        state["interests"] = data.get("interests", [])
    except Exception as e:
        print(f"Error extracting preferences: {e}")
        state["city"] = ""
        state["interests"] = []
        
    return state

def ask_for_city(state: PlannerState) -> PlannerState:
    """Fallback node if the destination city is missing."""
    state["itinerary"] = "I would love to help you plan your trip! However, I need to know your destination. Where would you like to go?"
    return state

def build_itinerary(state: PlannerState) -> PlannerState:
    """Generate the detailed travel itinerary."""
    prompt = ChatPromptTemplate.from_template(
        "You are an expert travel planner. Create a detailed, exciting 3-day itinerary for a trip to {city}. "
        "The traveler is particularly interested in: {interests}. "
        "Make sure to highlight activities, food, and sights that align perfectly with their interests. Keep it beautifully formatted."
    )
    
    interests_str = ", ".join(state["interests"]) if state["interests"] else "general sightseeing and popular attractions"
    response = llm.invoke(prompt.format(city=state["city"], interests=interests_str))
    
    state["itinerary"] = response.content
    return state

# ==========================================
# 3. Routing Logic
# ==========================================
def route_preferences(state: PlannerState) -> str:
    """Route based on whether a valid city was found."""
    if not state["city"] or state["city"].strip() == "":
        return "ask_for_city"
    return "build_itinerary"

# ==========================================
# 4. Graph Setup
# ==========================================
workflow = StateGraph(PlannerState)

# Add nodes
workflow.add_node("extract_preferences", extract_preferences)
workflow.add_node("ask_for_city", ask_for_city)
workflow.add_node("build_itinerary", build_itinerary)

# Add edges
workflow.add_edge(START, "extract_preferences")
workflow.add_conditional_edges(
    "extract_preferences",
    route_preferences,
    {
        "ask_for_city": "ask_for_city",
        "build_itinerary": "build_itinerary"
    }
)
workflow.add_edge("ask_for_city", END)
workflow.add_edge("build_itinerary", END)

# Compile
app = workflow.compile()

# ==========================================
# 5. Execution Wrapper
# ==========================================
def plan_travel(query: str) -> dict:
    """Invoke the travel planner LangGraph workflow."""
    initial_state = PlannerState(
        messages=[HumanMessage(content=query)],
        city="",
        interests=[],
        itinerary=""
    )
    result = app.invoke(initial_state)
    return result

# ==========================================
# 6. Command-line Interface
# ==========================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # CLI Argument execution
        user_query = " ".join(sys.argv[1:])
        print(f"Planning trip for: '{user_query}'\n")
        result = plan_travel(user_query)
        print("--- Travel Plan ---")
        print(f"Destination: {result['city'] if result['city'] else 'Missing'}")
        print(f"Interests:   {', '.join(result['interests']) if result['interests'] else 'General'}")
        print(f"\n{result['itinerary']}\n")
    else:
        # Interactive REPL execution
        print("🌍 Welcome to the Travel Planner Agent (Monolithic)!")
        print("Tell me where you want to go and what you like doing (e.g., 'I want to go to Kyoto and love museums and food').")
        print("Type 'exit' or 'quit' to stop.\n")
        
        while True:
            try:
                query = input("\nYour travel request: ")
                if query.lower() in ['exit', 'quit']:
                    print("Safe travels!")
                    break
                if not query.strip():
                    continue
                
                print("\nPlanning your trip... ✈️")
                result = plan_travel(query)
                print("\n--- Travel Plan ---")
                print(f"Destination: {result['city'] if result['city'] else 'Missing'}")
                print(f"Interests:   {', '.join(result['interests']) if result['interests'] else 'General'}")
                print(f"\n{result['itinerary']}")
                print("-" * 50)
            except KeyboardInterrupt:
                print("\nSafe travels!")
                break
