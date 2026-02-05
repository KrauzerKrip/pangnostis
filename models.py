from dataclasses import dataclass
import numpy as np

@dataclass
class Transcription:
    transcription: str
    summary: str
    filename: str
    embedding: np.ndarray