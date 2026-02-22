import json
import os
import yaml
from pathlib import Path

import matplotlib
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

from repository import TranscriptionRepository


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def get_embedding(text: str, client: genai.Client, repo: TranscriptionRepository) -> np.ndarray:
    """Retrieve embedding from DB cache or generate it."""
    # Check cache
    cached = repo.get_cached_embedding(text)
    if cached is not None:
        return cached

    # Generate
    res = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(task_type="CLUSTERING"),
    )
    vec = np.array(res.embeddings[0].values, dtype=np.float32)
    
    # Save cache
    repo.save_cached_embedding(text, vec)
    return vec


def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY missing.")
        return

    client = genai.Client(api_key=api_key)
    repo = TranscriptionRepository()
    
    # Load transcriptions
    transcriptions = repo.get_all()
    if not transcriptions:
        print("No transcriptions found in DB.")
        return

    # Extract abstract embeddings (since they are stored with transcription)
    # We will use abstract embeddings for this character match, as that's typical for broad meaning.
    # The prompt also mentioned "character of embeddings (of abstract and concrete summary)"
    # We can plot both abstract and concrete in the same graph, or just pick one. For clarity, let's use the abstract embedding (which is t.embedding).
    
    data_points = []
    for t in transcriptions:
        emb = t.embedding
        structured_data = {}
        try:
            structured_data = json.loads(t.structured_data)
        except:
            pass
            
        if isinstance(emb, bytes):
            emb = np.frombuffer(emb, dtype=np.float32)
        if isinstance(emb, np.ndarray) and emb.size > 0:
            data_points.append({
                "filename": t.filename,
                "embedding": normalize(emb),
                "summary": structured_data.get("abstract_summary", "No summary available")
            })

    if not data_points:
        print("No embeddings to process.")
        return

    # Load characteristics
    with open("config/characteristics.yaml", "r", encoding="utf-8") as f:
        characteristics_config = yaml.safe_load(f)

    spectrums = []

    for spectrum_name, sides in characteristics_config.items():
        side_vectors = {}
        for side_name, sentences in sides.items():
            vecs = []
            for s in sentences:
                vecs.append(get_embedding(s, client, repo))
            # Average vector
            mean_vec = np.mean(vecs, axis=0)
            side_vectors[side_name] = normalize(mean_vec)
        
        # Expect exactly two sides per spectrum
        side_names = list(side_vectors.keys())
        if len(side_names) != 2:
            print(f"Skipping {spectrum_name}, must have exactly 2 sides.")
            continue
            
        spectrums.append({
            "name": spectrum_name,
            "side_A_name": side_names[0],
            "side_A_vec": side_vectors[side_names[0]],
            "side_B_name": side_names[1],
            "side_B_vec": side_vectors[side_names[1]],
        })

    # Prepare plot
    num_spectrums = len(spectrums)
    fig, axes = plt.subplots(num_spectrums, 1, figsize=(12, 2 * num_spectrums), squeeze=False)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle("Character Spectrum Analysis (Abstract Summaries)", fontsize=16)

    colors = plt.colormaps["tab10"](np.linspace(0, 1, len(data_points)))
    markers = ["o", "s", "^", "v", "<", ">", "D", "p", "*", "h", "X", "8"]
    
    scatters = []

    for i, spec in enumerate(spectrums):
        ax = axes[i, 0]
        ax.set_title(spec["name"])
        
        # Draw a horizontal line for the axis
        ax.axhline(0, color='gray', linewidth=2)
        ax.set_yticks([])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_position(('data', 0))

        # Project and collect scores
        axis_vec = normalize(spec["side_A_vec"] - spec["side_B_vec"])
        scores = []
        for dp in data_points:
            score = np.dot(dp["embedding"], axis_vec)
            scores.append(score)
            
        # Dynamically scale X axis to fit the observed spread
        max_abs = max([abs(s) for s in scores]) if scores else 0.1
        if max_abs == 0:
            max_abs = 0.1
            
        x_limit = max_abs * 1.5  # Add 50% padding for visual clarity
        ax.set_xlim(-x_limit, x_limit)
        
        # Add labels for extremes at the dynamic limits
        ax.text(-x_limit, 0.1, f'← {spec["side_B_name"]}', horizontalalignment='left', fontsize=12, fontweight='bold')
        ax.text(x_limit, 0.1, f'{spec["side_A_name"]} →', horizontalalignment='right', fontsize=12, fontweight='bold')

        # Plot each transcription on this spectrum
        for j, score in enumerate(scores):
            # Format the summary to wrap lines
            summary_text = data_points[j]["summary"]
            wrapped_summary = "\n".join(
                [summary_text[k : k + 80] for k in range(0, len(summary_text), 80)]
            )
            
            s = ax.scatter(
                score, 0, 
                color=colors[j % len(colors)], 
                marker=markers[j % len(markers)], 
                s=150, 
                alpha=0.8,
                edgecolors="black",
                label=data_points[j]["filename"] if i == 0 else "" # Only add label once
            )
            s.metadata = (
                f"File: {data_points[j]['filename']}\n"
                f"Score: {score:.3f}\n"
                f"Summary:\n{wrapped_summary}"
            )
            scatters.append(s)

    # Add interactive tooltips
    cursor = mplcursors.cursor(scatters, hover=False)
    
    @cursor.connect("add")
    def on_add(sel):
        # Pull the metadata we stored earlier
        sel.annotation.set_text(sel.artist.metadata)
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9, boxstyle="round,pad=0.5")
        sel.annotation.set_clip_on(False)
        sel.annotation.set_zorder(100)

    plt.tight_layout()
    # Manually adjust the right margin to make room for the legend in the browser UI
    fig.subplots_adjust(right=0.75)
    if data_points:
        fig.legend(loc='center left', bbox_to_anchor=(0.77, 0.5), title="Transcriptions")
    
    out_dir = Path(".data/visual/character")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "character_spectrums.png"
    plt.savefig(out_path, bbox_inches='tight')
    
    print(f"Plot saved to {out_path}")
    
    try:
        plt.show()
    except Exception as e:
        print(f"Interactive plot skipped: {e}")


if __name__ == "__main__":
    main()
