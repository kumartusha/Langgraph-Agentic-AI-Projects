import json
import logging
import random
from datetime import datetime
from typing import Literal

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq

from src.models.domain_models import WeatherState
from src.services.weather_service import WeatherService
from src.services.email_service import EmailService
from src.config.settings import settings

logger = logging.getLogger(__name__)

class DisasterAgentService:
    def __init__(self):
        self.llm = ChatGroq(
            model=settings.LLM_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=0
        )
        self.app = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(WeatherState)

        workflow.add_node("get_weather", self.get_weather_data)
        workflow.add_node("social_media_monitoring", self.social_media_monitoring)
        workflow.add_node("analyze_disaster", self.analyze_disaster_type)
        workflow.add_node("assess_severity", self.assess_severity)
        workflow.add_node("data_logging", self.data_logging)
        workflow.add_node("emergency_response", self.emergency_response)
        workflow.add_node("civil_defense_response", self.civil_defense_response)
        workflow.add_node("public_works_response", self.public_works_response)
        workflow.add_node("get_human_verification", self.get_human_verification)
        workflow.add_node("send_email_alert", self.send_email_alert)
        workflow.add_node("handle_no_approval", self.handle_no_approval)

        workflow.add_edge("get_weather", "social_media_monitoring")
        workflow.add_edge("social_media_monitoring", "analyze_disaster")
        workflow.add_edge("analyze_disaster", "assess_severity")
        workflow.add_edge("assess_severity", "data_logging")
        workflow.add_conditional_edges("data_logging", self.route_response)
        workflow.add_edge("civil_defense_response", "get_human_verification")
        workflow.add_edge("public_works_response", "get_human_verification")
        workflow.add_conditional_edges("get_human_verification", self.verify_approval_router)
        workflow.add_edge("emergency_response", "send_email_alert")
        workflow.add_edge("send_email_alert", END)
        workflow.add_edge("handle_no_approval", END)

        workflow.set_entry_point("get_weather")
        return workflow.compile()

    def run_agent(self, city: str, simulate_weather: bool = False):
        initial_state = {
            "city": city,
            "simulate_weather": simulate_weather,
            "weather_data": {},
            "disaster_type": "",
            "severity": "",
            "response": "",
            "messages": [],
            "alerts": [],
            "social_media_reports": [],
            "human_approved": False
        }
        try:
            return self.app.invoke(initial_state)
        except Exception as e:
            logger.error(f"Error running agent for {city}: {str(e)}")
            raise e

    # ---------------------------------------------------------
    # Node Implementations
    # ---------------------------------------------------------
    def get_weather_data(self, state: WeatherState) -> WeatherState:
        city = state["city"]
        simulate = state.get("simulate_weather", False)
        
        weather_data = WeatherService.get_weather(city, simulate)
        state["weather_data"] = weather_data
        state["messages"].append(SystemMessage(content=f"Weather fetched for {city} (Simulated: {simulate})"))
        return state

    def social_media_monitoring(self, state: WeatherState) -> WeatherState:
        simulated_reports = [
            "Local reports of rising water levels and minor flooding.",
            "High winds causing power outages in parts of the city.",
            "Citizens reporting high temperatures and increased heat discomfort.",
            "Social media reports indicate severe storm damage in local infrastructure.",
            "Reports of traffic disruptions due to heavy rain.",
            "No unusual social media reports related to the weather at this time."
        ]
        report = random.choice(simulated_reports)
        state["social_media_reports"].append(report)
        state["messages"].append(SystemMessage(content=f"Social media report added: {report}"))
        return state

    def analyze_disaster_type(self, state: WeatherState) -> WeatherState:
        prompt = ChatPromptTemplate.from_template(
            "Based on the following weather conditions, identify if there's a potential weather disaster.\n"
            "Weather conditions:\n"
            "- Description: {weather}\n"
            "- Wind Speed: {wind_speed} m/s\n"
            "- Temperature: {temperature}°C\n"
            "- Humidity: {humidity}%\n"
            "- Pressure: {pressure} hPa\n"
            "Categorize into one of these exact types: Hurricane, Flood, Heatwave, Severe Storm, Winter Storm, or No Immediate Threat. Return ONLY the category name."
        )
        chain = prompt | self.llm
        result = chain.invoke(state["weather_data"]).content.strip()
        state["disaster_type"] = result
        state["messages"].append(SystemMessage(content=f"Disaster type identified: {result}"))
        return state

    def assess_severity(self, state: WeatherState) -> WeatherState:
        prompt = ChatPromptTemplate.from_template(
            "Given the weather conditions and identified disaster type '{disaster_type}', "
            "assess the severity level. Consider:\n"
            "- Weather: {weather}\n"
            "- Wind Speed: {wind_speed} m/s\n"
            "- Temperature: {temperature}°C\n"
            "Respond with ONLY one of these words: Critical, High, Medium, or Low"
        )
        chain = prompt | self.llm
        data = {**state["weather_data"], "disaster_type": state["disaster_type"]}
        severity = chain.invoke(data).content.strip()
        state["severity"] = severity
        state["messages"].append(SystemMessage(content=f"Severity assessed as: {severity}"))
        return state

    def emergency_response(self, state: WeatherState) -> WeatherState:
        prompt = ChatPromptTemplate.from_template(
            "Create an emergency response plan for a {disaster_type} situation "
            "with {severity} severity level in {city}. Include immediate actions needed."
        )
        chain = prompt | self.llm
        response = chain.invoke({"disaster_type": state["disaster_type"], "severity": state["severity"], "city": state["city"]}).content
        state["response"] = response
        state["messages"].append(SystemMessage(content="Emergency response plan generated"))
        return state

    def civil_defense_response(self, state: WeatherState) -> WeatherState:
        prompt = ChatPromptTemplate.from_template(
            "Create a civil defense response plan for a {disaster_type} situation "
            "with {severity} severity level in {city}. Focus on public safety measures."
        )
        chain = prompt | self.llm
        response = chain.invoke({"disaster_type": state["disaster_type"], "severity": state["severity"], "city": state["city"]}).content
        state["response"] = response
        state["messages"].append(SystemMessage(content="Civil defense response plan generated"))
        return state

    def public_works_response(self, state: WeatherState) -> WeatherState:
        prompt = ChatPromptTemplate.from_template(
            "Create a public works response plan for a {disaster_type} situation "
            "with {severity} severity level in {city}. Focus on infrastructure protection."
        )
        chain = prompt | self.llm
        response = chain.invoke({"disaster_type": state["disaster_type"], "severity": state["severity"], "city": state["city"]}).content
        state["response"] = response
        state["messages"].append(SystemMessage(content="Public works response plan generated"))
        return state

    def data_logging(self, state: WeatherState) -> WeatherState:
        # In a real app this would use a repository to save to a database.
        # We'll just log it.
        logger.info(f"DATA LOG: {state['city']} | {state['severity']} | {state['disaster_type']}")
        return state

    def get_human_verification(self, state: WeatherState) -> WeatherState:
        severity = state["severity"].strip().lower()
        if severity in ["low", "medium"]:
            # In a REST API, blocking `input()` is an anti-pattern.
            # We mock human approval as True here, but in real life this would be an async webhook.
            state["human_approved"] = True
            state["messages"].append(SystemMessage(content="Auto-mocked human approval to True for API compatibility."))
        else:
            state["human_approved"] = True
            state["messages"].append(SystemMessage(content=f"Auto-approved {severity} severity alert"))
        return state

    def send_email_alert(self, state: WeatherState) -> WeatherState:
        EmailService.send_alert(
            city=state["city"],
            severity=state["severity"],
            disaster_type=state["disaster_type"],
            weather_data=state["weather_data"],
            response_plan=state["response"],
            is_human_verified=state["human_approved"]
        )
        state["alerts"].append(f"Email alert sent: {datetime.now()}")
        state["messages"].append(SystemMessage(content="Successfully sent email alert"))
        return state

    def handle_no_approval(self, state: WeatherState) -> WeatherState:
        state["messages"].append(SystemMessage(content="Alert not sent due to lack of human approval"))
        return state

    # ---------------------------------------------------------
    # Routers
    # ---------------------------------------------------------
    def route_response(self, state: WeatherState) -> Literal["emergency_response", "send_email_alert", "civil_defense_response", "public_works_response"]:
        disaster = state["disaster_type"].strip().lower()
        severity = state["severity"].strip().lower()

        if severity in ["critical", "high"]:
            return "emergency_response"
        elif "flood" in disaster or "storm" in disaster:
            return "public_works_response"
        else:
            return "civil_defense_response"

    def verify_approval_router(self, state: WeatherState) -> Literal["send_email_alert", "handle_no_approval"]:
        return "send_email_alert" if state['human_approved'] else "handle_no_approval"
