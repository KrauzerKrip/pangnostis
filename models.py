from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class Transcription:
    transcription: str
    summary: str
    filename: str
    embedding: np.ndarray
    id: Optional[int] = None