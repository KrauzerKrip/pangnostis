from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class Transcription:
    transcription: str
    who: str
    what: str
    where: str
    context_vector_helper: str
    filename: str
    who_embedding: np.ndarray
    what_embedding: np.ndarray
    where_embedding: np.ndarray
    context_embedding: np.ndarray
    id: Optional[int] = None
