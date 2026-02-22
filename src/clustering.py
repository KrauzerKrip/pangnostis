import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import matplotlib
# Use WebAgg to open the interactive plot in a browser - very reliable on NixOS/Linux
try:
    matplotlib.use("WebAgg")
except ImportError:
    pass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import mplcursors
from dotenv import load_dotenv
from google import genai

from repository import TranscriptionRepository
from visualize import load_cache, save_cache, get_embedding


def get_next_visual_dir(base_dir: Path) -> Path:
    """Find the next incremental directory name in the base directory."""
    base_dir.mkdir(parents=True, exist_ok=True)
    existing_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if not existing_dirs:
        next_val = 1
    else:
        next_val = max(int(d.name) for d in existing_dirs) + 1

    new_dir = base_dir / str(next_val)
    new_dir.mkdir(parents=True, exist_ok=True)
    return new_dir


def do_clustering(
    embeddings: List[np.ndarray], 
    filenames: List[str], 
    summaries: List[str], 
    transcriptions: List[str],
    structured_datas: List[dict],
    title: str, 
    output_dir: Path, 
    emb_type: str
):
    num_transcriptions = len(embeddings)
    if num_transcriptions < 2:
         print(f"Not enough embeddings for {title} (minimum 2 required).")
         return
         
    embeddings_arr = np.array(embeddings)
    
    # PCA dimensionality reduction to 2D
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings_arr)
    
    # Number of clusters (limited by number of samples)
    n_clusters = min(5, num_transcriptions)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(reduced_embeddings)
    centroids = kmeans.cluster_centers_

    plt.figure(figsize=(14, 10))

    # Generate colors for clusters
    colors = plt.colormaps["tab10"](np.linspace(0, 1, n_clusters))

    # List of marker styles to cycle through for each file
    markers = ["o", "s", "^", "v", "<", ">", "D", "p", "*", "h", "X", "8"]

    scatters = []
    for i in range(num_transcriptions):
        s = plt.scatter(
            reduced_embeddings[i, 0],
            reduced_embeddings[i, 1],
            color=colors[labels[i]],
            marker=markers[i % len(markers)],
            label=filenames[i], # Only filename for the legend
            s=150,
            edgecolors="black",
            linewidths=0.5,
            alpha=0.85,
        )
        scatters.append(s)
        
        summary_text = summaries[i]
        wrapped_summary = "\n".join(
            [summary_text[j : j + 80] for j in range(0, len(summary_text), 80)]
        )
            
        s.metadata = (
            f"File: {filenames[i]}\n"
            f"Cluster: {labels[i]}\n"
            f"Summary:\n{wrapped_summary}"
        )

    # Add interactive tooltips
    cursor = mplcursors.cursor(scatters, hover=False)
    
    @cursor.connect("add")
    def on_add(sel):
        # Pull the metadata we stored earlier
        sel.annotation.set_text(sel.artist.metadata)
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9, boxstyle="round,pad=0.5")
        sel.annotation.set_clip_on(False)
        sel.annotation.set_zorder(100)

    # Plot centroids
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=200,
        linewidths=3,
        color="black",
        label="Cluster Centroids",
        zorder=10,
    )

    full_title = f"PCA Clustering: {title}"
    output_path = output_dir / f"clustering_{emb_type}.png"

    plt.title(full_title, fontsize=16)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)

    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        title="Files (Markers) & Clusters (Colors)",
        fontsize=8
    )

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    
    print(f"Opening interactive browser plot for '{emb_type}'...")
    try:
        plt.show()
    except Exception as e:
        print(f"Note: Could not open interactive plot window ({e}).")
    plt.close()
    
    print(f"Plot saved to {output_path}")

    # Export cluster data to JSON
    cluster_data = {f"cluster_{i}": [] for i in range(n_clusters)}

    for i in range(num_transcriptions):
        cluster_id = int(labels[i])
        cluster_data[f"cluster_{cluster_id}"].append({
            "filename": filenames[i],
            "cluster_id": cluster_id,
            "summary": summaries[i],
            "structured_data": structured_datas[i],
            "transcription": transcriptions[i]
        })

    json_output_path = output_dir / f"clustering_{emb_type}.json"
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(cluster_data, f, indent=4, ensure_ascii=False)
    print(f"Cluster data saved to {json_output_path}")


def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")

    client = genai.Client(api_key=api_key) if api_key else None
    repo = TranscriptionRepository()
    transcriptions = repo.get_all()

    if not transcriptions:
        print("No transcriptions found in the database.")
        return

    print(f"Found {len(transcriptions)} transcriptions.")

    abstract_data = [] # dt.dict
    concrete_data = [] # dt.dict

    # Cache for concrete embeddings
    cache_path = Path(".data/visual/concrete_embeddings_cache.pkl")
    embedding_cache = load_cache(cache_path)
    cache_updated = False

    print("Processing transcriptions...")
    for t in transcriptions:
        try:
            try:
                structured_data = json.loads(t.structured_data)
            except (json.JSONDecodeError, TypeError):
                print(f"Warning: structured_data is not valid JSON for {t.filename}. Skipping.")
                continue

            # 1. Abstract Summary
            abstract_text = structured_data.get("abstract_summary", "")
            if t.embedding is not None:
                emb = t.embedding
                if isinstance(emb, bytes):
                    emb = np.frombuffer(emb, dtype=np.float32)
                
                if isinstance(emb, np.ndarray) and emb.size > 0:
                    abstract_data.append({
                        "filename": t.filename,
                        "embedding": emb,
                        "summary": abstract_text,
                        "transcription": t.transcription,
                        "structured_data": structured_data
                    })
                else:
                    print(f"Warning: No valid embedding for {t.filename} (Abstract)")
            else:
                 print(f"Warning: No existing embedding for {t.filename} (Abstract)")

            # 2. Concrete Summary
            concrete_text = structured_data.get("concrete_summary", "")
            if concrete_text:
                if client:
                    emb = get_embedding(concrete_text, client, embedding_cache)
                    if emb is not None:
                        concrete_data.append({
                            "filename": t.filename,
                            "embedding": emb,
                            "summary": concrete_text,
                            "transcription": t.transcription,
                            "structured_data": structured_data
                        })
                        cache_updated = True
                else:
                     print(f"Skipping concrete embedding for {t.filename} (No API Key)")
            else:
                print(f"Warning: No concrete summary for {t.filename}")

        except Exception as e:
            print(f"Error processing {t.filename}: {e}")

    if cache_updated:
        save_cache(cache_path, embedding_cache)
        print("Updated embedding cache.")

    clustering_base_dir = Path(".data/visual/clustering")
    output_dir = get_next_visual_dir(clustering_base_dir)
    print(f"Output directory: {output_dir}")

    print("\n--- Clustering Abstract Summaries ---")
    if abstract_data:
        do_clustering(
            [d["embedding"] for d in abstract_data],
            [d["filename"] for d in abstract_data],
            [d["summary"] for d in abstract_data],
            [d["transcription"] for d in abstract_data],
            [d["structured_data"] for d in abstract_data],
            "Abstract Summaries",
            output_dir,
            "abstract"
        )
    else:
        print("No abstract data to cluster.")

    print("\n--- Clustering Concrete Summaries ---")
    if concrete_data:
        do_clustering(
            [d["embedding"] for d in concrete_data],
            [d["filename"] for d in concrete_data],
            [d["summary"] for d in concrete_data],
            [d["transcription"] for d in concrete_data],
            [d["structured_data"] for d in concrete_data],
            "Concrete Summaries",
            output_dir,
            "concrete"
        )
    else:
        print("No concrete data to cluster.")

if __name__ == "__main__":
    main()
