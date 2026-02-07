import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

from models import Transcription
from prompts import load_prompts
from repository import TranscriptionRepository


def get_transcription_text(json_path: Path) -> str:
    """Extract and concatenate text from the transcription JSON file."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        texts = [entry["text"] for entry in data.get("transcript", [])]
        return " ".join(texts)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return ""


def main():
    # Load environment variables from .env
    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found in environment. Please check your .env file.")
        return

    # Initialize Gemini client
    client = genai.Client(api_key=api_key)
    for m in client.models.list():
        print(f"Model: {m.name} supported actions:")
        for sa in m.supported_actions:
            print("    " + sa)
    # exit()
    # for model in client.models.list():
    #     model_info = json.loads(model.model_dump_json())
    #     print(model_info)
    # exit()
    # Initialize repository
    repo = TranscriptionRepository()

    # Load prompts
    try:
        prompts = load_prompts()
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return

    summarizer_prompt = prompts.get("summarizer")
    if not summarizer_prompt:
        print("Summarizer prompt not found in config/prompts.yaml")
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

    for json_file in json_files:
        print(f"--- Processing {json_file.name} ---")

        # Check if already processed to avoid redundant API calls
        existing = repo.get_by_filename(json_file.name)
        if existing:
            print(
                f"Skipping {json_file.name}, already in database (ID: {existing.id})."
            )
            continue

        transcription_text = get_transcription_text(json_file)
        if not transcription_text:
            print(f"No transcription text found in {json_file.name}")
            continue

        print(f"Generating summary for {json_file.name}...")
        try:
            # Generate Summary using the specified prompt
            # We'll use a word_count of 50 as a default for the template
            prompt_user = summarizer_prompt.user.format(
                filename=json_file.name,
                word_count=75,
                text=transcription_text,
                note="It's a transcription of a video replay where me and my friend play a video game. The filename has the name of the game.",
            )

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=summarizer_prompt.system
                ),
                contents=prompt_user,
            )
            summary = response.text
            if not summary:
                print(f"Failed to generate summary for {json_file.name}")
                continue

            print(f"Generating embedding for {json_file.name}...")
            # Generate Embeddings for the summary
            embed_response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=summary,
                config=types.EmbedContentConfig(task_type="CLUSTERING"),
            )

            # The response structure might vary slightly depending on the SDK version,
            # but usually it's in .embeddings[0].values
            embedding_values = embed_response.embeddings[0].values
            embedding = np.array(embedding_values, dtype=np.float32)

            # Create Transcription object
            transcription = Transcription(
                transcription=transcription_text,
                summary=summary,
                filename=json_file.name,
                embedding=embedding,
            )

            # Save to database
            repo.save(transcription)
            print(
                f"Successfully saved {json_file.name} to database (ID: {transcription.id})"
            )

        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")


if __name__ == "__main__":
    main()
