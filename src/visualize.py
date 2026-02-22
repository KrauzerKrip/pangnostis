import argparse
import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import matplotlib
# Use WebAgg for browser-based interactive plots (works well on headless/NixOS)
try:
    matplotlib.use("WebAgg")
except ImportError:
    pass

import matplotlib.pyplot as plt
import mplcursors
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sklearn.manifold import TSNE

from repository import TranscriptionRepository


def load_cache(cache_path: Path) -> Dict[str, np.ndarray]:
    """Load embedding cache from file."""
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cache {cache_path}: {e}")
    return {}


def save_cache(cache_path: Path, cache: Dict[str, np.ndarray]):
    """Save embedding cache to file."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
    except Exception as e:
        print(f"Error saving cache {cache_path}: {e}")


def get_embedding(
    text: str,
    client: genai.Client,
    cache: Dict[str, np.ndarray],
    model: str = "gemini-embedding-001",
) -> Optional[np.ndarray]:
    """Get embedding for text, using cache if available."""
    if not text:
        return None

    if text in cache:
        return cache[text]

    try:
        response = client.models.embed_content(
            model=model,
            contents=text,
            config=types.EmbedContentConfig(task_type="CLUSTERING"),
        )
        embedding = np.array(response.embeddings[0].values, dtype=np.float32)
        cache[text] = embedding
        return embedding
    except Exception as e:
        print(f"Error generating embedding for text '{text[:50]}...': {e}")
        return None


def plot_embeddings(
    embeddings: List[np.ndarray],
    filenames: List[str],
    summaries: List[str],
    display_summaries: List[str],
    title: str,
    output_path: Optional[Path] = None,
):
    """Perform t-SNE and plot with interactive tooltips."""
    if len(embeddings) < 2:
        print(f"Not enough embeddings to plot for {title} (minimum 2 required).")
        return

    # Convert to numpy array
    X = np.array(embeddings)
    
    # t-SNE dimensionality reduction
    perplexity = min(15, len(X) - 1)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    X_2d = tsne.fit_transform(X)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use a colormap
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(filenames)))
    
    # Scatter plot
    sc = ax.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=colors,
        s=100,
        alpha=0.8,
        edgecolors='w', 
        linewidth=0.5
    )

    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True, linestyle="--", alpha=0.6)

    # Add tooltips with mplcursors
    cursor = mplcursors.cursor(sc, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        index = sel.index
        filename = filenames[index]
        summary_text = display_summaries[index]
        
        # Format tooltip text
        wrapped_summary = "\n".join(
            [summary_text[i : i + 80] for i in range(0, len(summary_text), 80)]
        )
        sel.annotation.set_text(f"File: {filename}\nSummary:\n{wrapped_summary}")
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)


    plt.tight_layout()

    if output_path:
        # Save before showing, as show() might clear the figure or block
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")

    # Show plot interactively
    print("Opening interactive plot in browser...")
    try:
        plt.show()
    except Exception as e:
        print(f"\nNote: Could not open interactive plot window ({e}). Plots are saved to {output_path or 'disk'}.")



def main():
    parser = argparse.ArgumentParser(description="Visualize transcription embeddings.")
    parser.add_argument("--save", action="store_true", help="Save plots to .data/visual")
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    # if not api_key:
    #     print("GEMINI_API_KEY not found in environment. Needed for generating concrete embeddings.")
    #     return

    client = genai.Client(api_key=api_key) if api_key else None
    repo = TranscriptionRepository()
    transcriptions = repo.get_all()

    if not transcriptions:
        print("No transcriptions found in the database.")
        return

    print(f"Found {len(transcriptions)} transcriptions.")

    # Prepare data containers
    # We will hold lists of tuples: (transcription, abstract_emb, concrete_emb_or_text)
    
    abstract_data = [] # (filename, embedding, summary_text)
    concrete_data = [] # (filename, embedding, summary_text)

    # Cache for concrete embeddings
    cache_path = Path(".data/visual/concrete_embeddings_cache.pkl")
    embedding_cache = load_cache(cache_path)
    cache_updated = False

    print("Processing transcriptions...")
    for t in transcriptions:
        try:
            # Check if structured_data is valid JSON
            try:
                structured_data = json.loads(t.structured_data)
            except (json.JSONDecodeError, TypeError):
                print(f"Warning: structured_data is not valid JSON for {t.filename}. Skipping.")
                continue

            # 1. Abstract Summary (uses existing embedding if valid)
            abstract_text = structured_data.get("abstract_summary", "")
            if t.embedding is not None:
                # Need to convert bytes to numpy array if it's stored as bytes in DB object, 
                # but models.py says np.ndarray. Assuming repository handles conversion.
                # If t.embedding is bytes, we might need: np.frombuffer(t.embedding, dtype=np.float32)
                # However, previous visualize.py code used getattr(t, attr_name) directly.
                
                # Let's verify type. If it's bytes, convert.
                emb = t.embedding
                if isinstance(emb, bytes):
                    emb = np.frombuffer(emb, dtype=np.float32)
                
                if isinstance(emb, np.ndarray) and emb.size > 0:
                    abstract_data.append({
                        "filename": t.filename,
                        "embedding": emb,
                        "summary": abstract_text
                    })
                else:
                    print(f"Warning: No valid embedding for {t.filename} (Abstract)")
            else:
                 print(f"Warning: No existing embedding for {t.filename} (Abstract)")

            # 2. Concrete Summary (needs new embedding)
            concrete_text = structured_data.get("concrete_summary", "")
            if concrete_text:
                if client:
                    emb = get_embedding(concrete_text, client, embedding_cache)
                    if emb is not None:
                        concrete_data.append({
                            "filename": t.filename,
                            "embedding": emb,
                            "summary": concrete_text
                        })
                        cache_updated = True
                else:
                     print(f"Skipping concrete embedding for {t.filename} (No API Key)")
            else:
                print(f"Warning: No concrete summary for {t.filename}")

        except Exception as e:
            print(f"Error processing {t.filename}: {e}")

    # Save cache if updated
    if cache_updated:
        save_cache(cache_path, embedding_cache)
        print("Updated embedding cache.")

    # Setup output directory if saving
    output_dir = None
    if args.save:
        visual_base_dir = Path(".data/visual")
        visual_base_dir.mkdir(parents=True, exist_ok=True)
        existing_dirs = [d for d in visual_base_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        next_val = 1 if not existing_dirs else max(int(d.name) for d in existing_dirs) + 1
        output_dir = visual_base_dir / str(next_val)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving visualizations to {output_dir}")

    # Plot Abstract Summaries
    print("\n--- Plotting Abstract Summaries ---")
    if abstract_data:
        plot_embeddings(
            [d["embedding"] for d in abstract_data],
            [d["filename"] for d in abstract_data],
            [d["summary"] for d in abstract_data], 
            [d["summary"] for d in abstract_data],
            "Abstract Summary Clustering",
            output_path=output_dir / "abstract_clustering.png" if output_dir else None
        )
    else:
        print("No abstract data to plot.")

    # Plot Concrete Summaries
    print("\n--- Plotting Concrete Summaries ---")
    if concrete_data:
        plot_embeddings(
            [d["embedding"] for d in concrete_data],
            [d["filename"] for d in concrete_data],
            [d["summary"] for d in concrete_data],
            [d["summary"] for d in concrete_data],
            "Concrete Summary Clustering",
            output_path=output_dir / "concrete_clustering.png" if output_dir else None
        )
    else:
        print("No concrete data to plot.")

if __name__ == "__main__":
    main()
