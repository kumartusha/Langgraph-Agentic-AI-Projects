from fastapi import APIRouter
from src.controllers import analysis_controller

# Main API Router
api_router = APIRouter()

# Mount the analysis controller routes
api_router.include_router(analysis_controller.router, tags=["Audio Analysis"])
