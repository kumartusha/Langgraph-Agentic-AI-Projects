from fastapi import APIRouter, UploadFile, File, HTTPException
from src.services.transcription_service import TranscriptionService
from src.services.crewai_service import CrewAIService
from src.repositories.analysis_repository import AnalysisRepository
from src.models.domain_models import AnalysisResponse
from src.utils.file_helpers import save_upload_file_tmp, cleanup_tmp_file
from src.config.settings import settings

router = APIRouter()
transcription_service = TranscriptionService()
analysis_repo = AnalysisRepository()

@router.post("/analyze/audio", response_model=AnalysisResponse)
def analyze_audio(audio_file: UploadFile = File(...)):
    """
    Endpoint to upload an audio file, transcribe it, and extract CRM insights using AI.
    """
    if not audio_file.filename.endswith(('.mp3', '.wav', '.m4a')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only mp3, wav, and m4a are supported.")

    tmp_path = None
    try:
        # 1. Save File
        tmp_path = save_upload_file_tmp(audio_file)

        # 2. Transcribe Audio
        transcription = transcription_service.transcribe_audio(tmp_path)
        
        if not transcription.strip():
            raise HTTPException(status_code=400, detail="Transcription failed or audio was empty.")

        # 3. Analyze Transcription using CrewAI
        # Re-initialize with key in case it was loaded late
        import os
        api_key = settings.GROQ_API_KEY or settings.GROK_API_KEY or os.getenv("GROK_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GROK_API_KEY is missing.")
            
        crew_service = CrewAIService(api_key=api_key)
        analysis_result = crew_service.analyze_transcription(transcription)

        # 4. Save to Repository (Mock DB)
        analysis_repo.save_analysis(audio_file.filename, transcription, analysis_result)

        # 5. Return JSON Response
        return AnalysisResponse(
            filename=audio_file.filename,
            transcription=transcription,
            analysis=analysis_result
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if tmp_path:
            cleanup_tmp_file(tmp_path)
