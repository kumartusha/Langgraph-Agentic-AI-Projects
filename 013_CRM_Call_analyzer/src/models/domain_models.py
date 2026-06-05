from typing import List, Optional
from pydantic import BaseModel, Field
from src.constants.enums import CallSentiment

class ActionableInsight(BaseModel):
    action: str = Field(..., description="Describe specific actions that can be taken to address the customer's concerns or enhance the sales process.")
    assigned_to: str = Field(..., description="Specify the individual or team responsible for the action.")
    timeline: str = Field(..., description="Provide a timeframe or deadline for completing the action.")

class CallAnalysisResult(BaseModel):
    """
    Structured Output schema for the CrewAI Analysis Task.
    """
    sentiment_analysis: str = Field(..., description="Describe the sentiment of the customer interaction, including any notable emotional tone or reservations.")
    key_phrases: List[str] = Field(..., description="List key phrases that capture the customer's main interests, concerns, or preferences.")
    customer_pain_points: List[str] = Field(..., description="List specific challenges, objections, or concerns raised by the customer during the conversation.")
    agent_effectiveness_score: int = Field(..., description="Provide a rating or score out of 10 based on the agent's performance, including aspects like communication, problem-solving, and empathy.", ge=0, le=10)
    sales_opportunities: List[str] = Field(..., description="Identify potential sales opportunities based on the conversation, such as upselling or cross-selling.")
    competitor_mentions: str = Field(..., description="Mention any competitors that were brought up by the customer and relevant context. If none, state 'No relevant information found'.")
    call_engagement: str = Field(..., description="Describe the level of customer engagement during the call, noting any periods of silence, hesitation, or active discussion.")
    recommendations: str = Field(..., description="Provide strategic recommendations based on the analysis of the conversation, aimed at improving the interaction or future sales success.")
    actionable_insights: List[ActionableInsight] = Field(..., description="Clear next steps for both the agent and the customer, with assigned responsibilities and timelines.")

class AnalysisResponse(BaseModel):
    """
    Final HTTP JSON Response.
    """
    filename: str
    transcription: str
    analysis: CallAnalysisResult
