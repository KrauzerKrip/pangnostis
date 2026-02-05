from pydantic import BaseModel

class Transcription(BaseModel):
    transcription: str
    summary: str
    filename: str
    embedding: list[float]

