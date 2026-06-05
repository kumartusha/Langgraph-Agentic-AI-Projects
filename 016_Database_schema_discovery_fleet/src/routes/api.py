from fastapi import APIRouter
from src.controllers import chat_controller

router = APIRouter()

router.include_router(chat_controller.router, prefix="/v1", tags=["Chat"])
