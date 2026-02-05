# Project Context

- **Package Manager:** `uv`
  - Add dependencies: `uv add <package>` && `uv sync`
  - Run project: `uv run <command>` (e.g., `uv run main.py`)
- **Functionality:**
  - Uses LLMs via API.
  - Reads text data (transcriptions) from `.data` directory.
  - Processes the transcribed text.

## Implemented Components

### Prompt System (`prompts.py`)
- **Class `Prompt`**: Dataclass holding `name`, `system`, and `user` templates.
- **Utility `load_prompts()`**: Loads configurations from `config/prompts.yaml` into a dictionary of `Prompt` objects.

### Data Models (`models.py`)
- **Class `Transcription`**: Standard Python dataclass.
  - `transcription`: Raw text (string).
  - `summary`: Processed summary (string).
  - `filename`: Source filename (string).
  - `embedding`: Summary vector representation (`numpy.ndarray` of float32).

### Repository (`repository.py`)
- **Class `TranscriptionRepository`**: Manages persistence in SQLite (`.data/pangnostis.db`).
- **Vector Search**: Integrated `sqlite-vec` extension for efficient embedding storage and retrieval.
- **Storage Format**: Embeddings are stored as binary BLOBs (Little-endian Float32) compatible with `sqlite-vec`.

### Environment & Dependencies
- **Nix Flake**: Updated to provide `numpy` and `LD_LIBRARY_PATH` (for `libstdc++` compatibility on NixOS).
- **Python Dependencies**: `PyYAML`, `sqlite-vec`, `pydantic`.