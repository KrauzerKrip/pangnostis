import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from pathlib import Path
from repository import TranscriptionRepository


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


def plot_embeddings(embeddings, filenames, title, output_path):
    """Perform t-SNE and save a scatter plot."""
    # t-SNE dimensionality reduction
    perplexity = min(30, len(embeddings) - 1)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plotting
    plt.figure(figsize=(12, 8))
    colors = plt.colormaps["tab10"](np.linspace(0, 1, len(filenames)))

    for i, filename in enumerate(filenames):
        plt.scatter(
            embeddings_2d[i, 0],
            embeddings_2d[i, 1],
            color=colors[i],
            label=filename,
            s=100,
        )

    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Transcriptions")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    repo = TranscriptionRepository()
    transcriptions = repo.get_all()

    if not transcriptions:
        print("No transcriptions found in the database.")
        return

    if len(transcriptions) < 2:
        print("Need at least 2 transcriptions to perform t-SNE.")
        return

    # Define embedding types and their corresponding attribute names in the Transcription model
    embedding_map = {
        "who": "who_embedding",
        "what": "what_embedding",
        "when": "when_embedding",
        "where": "where_embedding",
        "context": "context_embedding"
    }

    # Setup output directory
    visual_base_dir = Path(".data/visual")
    output_dir = get_next_visual_dir(visual_base_dir)
    print(f"Saving visualizations to {output_dir}")

    filenames = [t.filename for t in transcriptions]

    for emb_type, attr_name in embedding_map.items():
        print(f"Processing '{emb_type}' embeddings...")
        embeddings = np.array([getattr(t, attr_name) for t in transcriptions])
        
        output_path = output_dir / f"embedding_{emb_type}.png"
        title = f"t-SNE Visualization: {emb_type.capitalize()} Embeddings"
        
        plot_embeddings(embeddings, filenames, title, output_path)

    print("\nVisualization complete for all embedding types.")


if __name__ == "__main__":
    main()