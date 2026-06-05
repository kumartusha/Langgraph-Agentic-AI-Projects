from fastapi import APIRouter
from src.controllers import code_controller

api_router = APIRouter()

# Register our code healing controller
api_router.include_router(code_controller.router, prefix="/code", tags=["Self Healing Code"])
