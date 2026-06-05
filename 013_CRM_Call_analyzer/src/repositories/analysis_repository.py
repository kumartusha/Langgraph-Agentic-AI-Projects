import json
from typing import Dict, Any
from src.models.domain_models import CallAnalysisResult

class AnalysisRepository:
    """
    Mock Data Access Layer.
    Saves analysis results to local JSON files instead of a real database.
    """
    def __init__(self):
        self.db_path = "db_mock"
        import os
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)

    def save_analysis(self, filename: str, transcription: str, analysis: CallAnalysisResult) -> Dict[str, Any]:
        """
        Saves the analysis to disk.
        """
        record = {
            "filename": filename,
            "transcription": transcription,
            "analysis": analysis.model_dump()
        }
        
        filepath = f"{self.db_path}/{filename}_analysis.json"
        with open(filepath, "w") as f:
            json.dump(record, f, indent=4)
            
        return record
