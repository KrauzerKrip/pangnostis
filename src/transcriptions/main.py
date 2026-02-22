import os
import json
from pathlib import Path
from faster_whisper import WhisperModel

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_AUDIO_DIR = BASE_DIR / ".data" / "audio" / "raw"
TRANSCRIPTIONS_DIR = BASE_DIR / ".data" / "audio" / "transcriptions"

# Whisper settings
MODEL_SIZE = "large-v3" # You can change to large-v2 or medium based on speed/quality needs
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

def transcribe_file(model: WhisperModel, audio_path: Path, output_path: Path):
    try:
        segments, info = model.transcribe(str(audio_path), beam_size=5)

        print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")

        transcription_data = {
            "source_file": audio_path.name,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "segments": []
        }

        # List to hold text for console output if needed
        for segment in segments:
            segment_data = {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            }
            transcription_data["segments"].append(segment_data)

        print(f"Finished transcribing {len(transcription_data['segments'])} segments.")

        # Save to JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcription_data, f, ensure_ascii=False, indent=2)

        print(f"Saved: {output_path.relative_to(BASE_DIR)}")

    except Exception as e:
        print(f"Error transcribing {audio_path.name}: {e}")

def process_audio_files():
    if not RAW_AUDIO_DIR.exists():
        print(f"Raw audio directory not found: {RAW_AUDIO_DIR}")
        return

    # Ensure output directory exists
    TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)

    model = None

    # Traverse the raw audio directories
    for session_dir in RAW_AUDIO_DIR.iterdir():
        if not session_dir.is_dir():
            continue

        session_name = session_dir.name
        output_session_dir = TRANSCRIPTIONS_DIR / session_name
        output_session_dir.mkdir(parents=True, exist_ok=True)

        # Process each mp3 file in the session directory
        for audio_file in session_dir.glob("*.mp3"):
            if "combined" in audio_file.name.lower():
                print(f"Skipping combined recording: {audio_file.name}")
                continue

            output_file = output_session_dir / f"{audio_file.stem}.json"

            if output_file.exists():
                print(f"Transcription already exists, skipping: {output_file.relative_to(BASE_DIR)}")
                continue

            # Lazy load model only if we need to transcribe something
            if model is None:
                print(f"Loading faster-whisper model '{MODEL_SIZE}' on {DEVICE} ({COMPUTE_TYPE})...")
                model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
                print("Model loaded successfully.")

            print(f"Transcribing: {audio_file.relative_to(BASE_DIR)}")
            transcribe_file(model, audio_file, output_file)

if __name__ == "__main__":
    process_audio_files()
