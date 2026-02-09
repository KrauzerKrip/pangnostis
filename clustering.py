from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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


def main():
    repo = TranscriptionRepository()
    transcriptions = repo.get_all()

    if not transcriptions:
        print("No transcriptions found in the database.")
        return

    num_transcriptions = len(transcriptions)
    if num_transcriptions < 2:
        print("Need at least 2 transcriptions to perform clustering.")
        return

    # Define embedding types and their corresponding attribute names in the Transcription model
    embedding_map = {
        "who": "who_embedding",
        "what": "what_embedding",
        "where": "where_embedding",
        "context": "context_embedding",
    }

    clustering_base_dir = Path(".data/visual/clustering")
    output_dir = get_next_visual_dir(clustering_base_dir)
    filenames = [t.filename for t in transcriptions]

    # List of marker styles to cycle through for each file
    markers = ["o", "s", "^", "v", "<", ">", "D", "p", "*", "h", "X", "8"]

    # Number of clusters (limited by number of samples)
    n_clusters = min(5, num_transcriptions)

    for emb_type, attr_name in embedding_map.items():
        print(f"Processing '{emb_type}' embeddings...")
        embeddings = np.array([getattr(t, attr_name) for t in transcriptions])

        # PCA dimensionality reduction to 2D
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)

        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        # labels = kmeans.fit_predict(
        #     embeddings
        # )  # Cluster on full embeddings or reduced? Usually full is better.
        # centroids = kmeans.cluster_centers_

        # Transform centroids for plotting if they were calculated on full embeddings
        # Actually, let's cluster on reduced embeddings for visual consistency with the plot
        # or just cluster on full and map centroids.
        # For simplicity and visual "tightness", clustering on reduced embeddings works well for 2D plots.
        labels = kmeans.fit_predict(reduced_embeddings)
        centroids = kmeans.cluster_centers_

        plt.figure(figsize=(14, 10))

        # Generate colors for clusters
        colors = plt.colormaps["tab10"](np.linspace(0, 1, n_clusters))

        # Plot each transcription point
        for i in range(num_transcriptions):
            plt.scatter(
                reduced_embeddings[i, 0],
                reduced_embeddings[i, 1],
                color=colors[labels[i]],
                marker=markers[i % len(markers)],
                label=filenames[i],
                s=150,
                edgecolors="black",
                linewidths=0.5,
                alpha=0.85,
            )

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

        title = f"PCA Clustering: {emb_type.capitalize()} Embeddings"
        output_path = output_dir / f"clustering_{emb_type}.png"

        plt.title(title, fontsize=16)
        plt.xlabel("Principal Component 1", fontsize=12)
        plt.ylabel("Principal Component 2", fontsize=12)

        # Legend: filenames with their respective markers and cluster colors
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            title="Files (Markers) & Clusters (Colors)",
            fontsize=10,
        )

        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Plot saved to {output_path}")

    print(f"\nVisualization complete. All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
