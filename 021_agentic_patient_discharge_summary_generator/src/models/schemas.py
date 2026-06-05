from pydantic import BaseModel, Field
from typing import List, Optional

class Medication(BaseModel):
    name: str = Field(description="The exact name of the medication")
    dosage: Optional[str] = Field(description="The dosage of the medication, if provided (e.g., 500mg)", default=None)
    frequency: Optional[str] = Field(description="How often the medication is taken (e.g., BID, daily)", default=None)

class Diagnosis(BaseModel):
    name: str = Field(description="The medical diagnosis or condition")
    status: Optional[str] = Field(description="Status of diagnosis e.g. chronic, acute, resolved", default=None)

class ExtractedEntities(BaseModel):
    """Schema for extracting medical entities from unstructured clinical text."""
    diagnoses: List[Diagnosis] = Field(description="List of all diagnoses found in the note", default_factory=list)
    medications: List[Medication] = Field(description="List of all medications (admission or discharge) found in the note", default_factory=list)
    procedures: List[str] = Field(description="List of any medical procedures mentioned", default_factory=list)
