from fastapi import APIRouter
from src.controllers.disaster_controller import router as disaster_router

api_router = APIRouter()
api_router.include_router(disaster_router, prefix="/disaster", tags=["Disaster Management Agent"])
