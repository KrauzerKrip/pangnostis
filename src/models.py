from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class Transcription:
    transcription: str
    structured_data: str  # JSON string
    filename: str
    embedding: np.ndarray
    id: Optional[int] = None
