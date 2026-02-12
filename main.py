import argparse
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from models import Transcription
from prompts import load_prompts
from repository import TranscriptionRepository


class StructuredSummary(BaseModel):
    who: str
    what: str
    where: str
    context_vector_helper: str


def extract_datetime(filename: str) -> datetime:
    """Extract datetime from filename using regex patterns."""
    # Pattern 1: YYYY-MM-DD HH-MM-SS
    match1 = re.search(r"(\d{4})-(\d{2})-(\d{2}) (\d{2})-(\d{2})-(\d{2})", filename)
    if match1:
        return datetime.strptime(match1.group(0), "%Y-%m-%d %H-%M-%S")

    # Pattern 2: YYYY.MM.DD - HH.MM.SS
    match2 = re.search(
        r"(\d{4})\.(\d{2})\.(\d{2}) - (\d{2})\.(\d{2})\.(\d{2})", filename
    )
    if match2:
        return datetime.strptime(match2.group(0), "%Y.%m.%d - %H.%M.%S")

    return None


def get_transcription_text(json_path: Path, include_time: bool = False, time_offset: float = 0.0) -> str:
    """Extract and concatenate text from the transcription JSON file."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        prefix = ""
        if include_time:
            dt = extract_datetime(json_path.name)
            if dt:
                prefix = f"[Recorded at {dt.strftime('%Y-%m-%d %H:%M:%S')}]\n"

        lines = []
        for entry in data.get("transcript", []):
            start = entry.get("start", 0.0) + time_offset
            end = entry.get("end", 0.0) + time_offset
            text = entry.get("text", "").strip()
            lines.append(f"[{start:.2f}s-{end:.2f}s] {text}")
            
        return prefix + "\n".join(lines)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return ""


def main():
    parser = argparse.ArgumentParser(
        description="Process transcriptions and generate summaries."
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge transcriptions recorded within 1 minute of each other.",
    )
    args = parser.parse_args()

    # Load environment variables from .env
    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found in environment. Please check your .env file.")
        return

    # Initialize Gemini client
    client = genai.Client(api_key=api_key)

    # Initialize repository
    repo = TranscriptionRepository()

    # Load prompts
    try:
        prompts = load_prompts()
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return

    summarizer_prompt = prompts.get("structured_summarizer")
    if not summarizer_prompt:
        print("structured_summarizer prompt not found in config/prompts.yaml")
        return

    workspace_dir = Path(".data/workspace")
    if not workspace_dir.exists():
        print(f"Workspace directory {workspace_dir} does not exist.")
        return

    # Process each JSON file in the workspace
    json_files = list(workspace_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {workspace_dir}")
        return

    tasks = []
    if args.merge:
        # Group files by time
        file_datetimes = []
        for f in json_files:
            dt = extract_datetime(f.name)
            if dt:
                file_datetimes.append((f, dt))
            else:
                # If no datetime found, treat as standalone task
                tasks.append([f])

        # Sort by datetime
        file_datetimes.sort(key=lambda x: x[1])

        if file_datetimes:
            current_group = [file_datetimes[0][0]]
            last_dt = file_datetimes[0][1]

            for i in range(1, len(file_datetimes)):
                f, dt = file_datetimes[i]
                if dt - last_dt <= timedelta(minutes=1):
                    current_group.append(f)
                else:
                    tasks.append(current_group)
                    current_group = [f]
                last_dt = dt
            tasks.append(current_group)
    else:
        # Each file is its own task
        tasks = [[f] for f in json_files]

    for task_files in tasks:
        task_filename = ";".join([f.name for f in task_files])
        display_name = (
            task_files[0].name
            if len(task_files) == 1
            else f"Merged ({len(task_files)} files)"
        )

        print(f"--- Processing {display_name} ---")

        # Check if already processed to avoid redundant API calls
        existing = repo.get_by_filename(task_filename)
        if existing:
            print(f"Skipping {display_name}, already in database (ID: {existing.id}).")
            continue

        # Combine transcriptions
        transcription_parts = []
        base_dt = extract_datetime(task_files[0].name) if task_files else None

        for f in task_files:
            offset = 0.0
            if base_dt:
                current_dt = extract_datetime(f.name)
                if current_dt:
                    offset = (current_dt - base_dt).total_seconds()

            part = get_transcription_text(f, include_time=args.merge, time_offset=offset)
            if part:
                transcription_parts.append(part)

        joined_transcription_text = "\n\n".join(transcription_parts)

        if not joined_transcription_text:
            print(f"No transcription text found for {display_name}")
            continue

        print(f"Generating structured summary for {display_name}...")
        try:
            # Generate Summary using the specified prompt
            prompt_user = summarizer_prompt.user.format(
                filename=task_filename,
                word_count=150,
                text=joined_transcription_text,
                note="It's a transcription of a video replay where me and my friend play a video game. The filename has the name of the game.",
            )

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=summarizer_prompt.system,
                    response_mime_type="application/json",
                    response_schema=StructuredSummary,
                ),
                contents=prompt_user,
            )

            try:
                # The SDK might return the object directly if response_schema is used,
                # but usually we still need to parse the text or use response.parsed
                if hasattr(response, "parsed") and response.parsed:
                    structured_data_obj = response.parsed
                    structured_data = structured_data_obj.model_dump()
                else:
                    structured_data = json.loads(response.text)
            except Exception as e:
                print(f"Failed to parse response for {display_name}: {e}")
                print(f"Raw response: {response.text}")
                continue

            # Fields: who, what, where, context_vector_helper
            fields = ["who", "what", "where", "context_vector_helper"]
            embeddings = {}

            print(f"Generating embeddings for {display_name}...")
            for field in fields:
                field_content = structured_data.get(field, "")
                if not field_content:
                    print(f"Warning: Field '{field}' is empty for {display_name}")

                embed_response = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=field_content or "None",
                    config=types.EmbedContentConfig(task_type="CLUSTERING"),
                )
                embeddings[field] = np.array(
                    embed_response.embeddings[0].values, dtype=np.float32
                )

            # Create Transcription object
            transcription = Transcription(
                transcription=joined_transcription_text,
                who=structured_data.get("who", ""),
                what=structured_data.get("what", ""),
                where=structured_data.get("where", ""),
                context_vector_helper=structured_data.get("context_vector_helper", ""),
                filename=task_filename,
                who_embedding=embeddings["who"],
                what_embedding=embeddings["what"],
                where_embedding=embeddings["where"],
                context_embedding=embeddings["context_vector_helper"],
            )

            # Save to database
            repo.save(transcription)
            print(
                f"Successfully saved {display_name} to database (ID: {transcription.id})"
            )

        except Exception as e:
            print(f"Error processing {display_name}: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
