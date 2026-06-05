from fastapi import APIRouter, HTTPException
from src.models.domain_models import DisasterAnalysisRequest, DisasterAnalysisResponse
from src.services.disaster_agent_service import DisasterAgentService

router = APIRouter()
agent_service = DisasterAgentService()

@router.post("/analyze", response_model=DisasterAnalysisResponse)
def analyze_disaster(request: DisasterAnalysisRequest):
    """
    Trigger the Weather Disaster Management LangGraph workflow for a specific city.
    """
    try:
        # Note: LangGraph's invoke is sync by default unless async is explicitly configured.
        # So we use a standard synchronous def for the FastAPI route to avoid event loop conflicts.
        result = agent_service.run_agent(city=request.city, simulate_weather=request.simulate_weather)
        
        return DisasterAnalysisResponse(
            city=result.get("city", request.city),
            weather_data=result.get("weather_data", {}),
            disaster_type=result.get("disaster_type", "Unknown"),
            severity=result.get("severity", "Unknown"),
            response_plan=result.get("response", "No response generated"),
            alerts_sent=result.get("alerts", []),
            human_approved=result.get("human_approved", False)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
