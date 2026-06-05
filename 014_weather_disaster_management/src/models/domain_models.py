from pydantic import BaseModel, Field
from typing import Dict, List, Union, Optional
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ---------------------------------------------------------
# API Request / Response DTOs
# ---------------------------------------------------------

class DisasterAnalysisRequest(BaseModel):
    city: str = Field(..., description="The name of the city to monitor and analyze")
    simulate_weather: bool = Field(False, description="If true, use simulated high-severity weather instead of live API data")

class WeatherDataModel(BaseModel):
    weather: str
    wind_speed: Union[str, float, int]
    cloud_cover: Union[str, float, int]
    sea_level: Union[str, float, int]
    temperature: Union[str, float, int]
    humidity: Union[str, float, int]
    pressure: Union[str, float, int]

class DisasterAnalysisResponse(BaseModel):
    city: str
    weather_data: WeatherDataModel
    disaster_type: str
    severity: str
    response_plan: str
    alerts_sent: List[str]
    human_approved: bool

# ---------------------------------------------------------
# LangGraph State Definitions
# ---------------------------------------------------------

class WeatherState(TypedDict):
    city: str
    simulate_weather: bool
    weather_data: Dict
    disaster_type: str
    severity: str
    response: str
    messages: List[Union[SystemMessage, HumanMessage, AIMessage]]
    alerts: List[str]
    social_media_reports: List[str]
    human_approved: bool
