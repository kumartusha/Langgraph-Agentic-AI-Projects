import os
from groq import Groq
from src.config.settings import settings

class TranscriptionService:
    def __init__(self):
        api_key = settings.GROQ_API_KEY or settings.GROK_API_KEY or os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
        if not api_key:
            raise ValueError("Groq API key not found. Please set GROK_API_KEY in your .env")
            
        self.client = Groq(api_key=api_key)

    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio from a given file path using Groq's whisper-large-v3 model.
        """
        try:
            with open(audio_path, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=(os.path.basename(audio_path), file.read()),
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json",  
                )
            return transcription.text
        except Exception as e:
            raise RuntimeError(f"Error during transcription: {str(e)}")
