import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from repository import TranscriptionRepository


def main():
    repo = TranscriptionRepository()
    transcriptions = repo.get_all()

    if not transcriptions:
        print("No transcriptions found in the database.")
        return

    if len(transcriptions) < 2:
        print("Need at least 2 transcriptions to perform t-SNE.")
        return

    # Extract embeddings and filenames
    embeddings = np.array([t.embedding for t in transcriptions])
    filenames = [t.filename for t in transcriptions]

    print(f"Performing t-SNE on {len(embeddings)} embeddings...")

    # t-SNE dimensionality reduction
    # perplexity should be less than the number of samples
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

    # Use a colormap to get distinct colors
    colors = plt.colormaps["tab10"](np.linspace(0, 1, len(filenames)))

    for i, filename in enumerate(filenames):
        plt.scatter(
            embeddings_2d[i, 0],
            embeddings_2d[i, 1],
            color=colors[i],
            label=filename,
            s=100,
        )

    plt.title("t-SNE Visualization of Transcription Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Transcriptions")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    output_path = "embeddings_plot.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

    # Find 4 closest to each other
    if len(embeddings) >= 4:
        from sklearn.metrics import pairwise_distances

        dist_matrix = pairwise_distances(embeddings, metric="cosine")

        min_dist_sum = float("inf")
        best_indices = []

        for i in range(len(embeddings)):
            # Find 3 nearest neighbors of i (including i itself)
            neighbor_indices = np.argsort(dist_matrix[i])[:4]

            # Compute total pairwise distance among these 4
            current_dist_sum = 0
            for idx1 in range(4):
                for idx2 in range(idx1 + 1, 4):
                    current_dist_sum += dist_matrix[
                        neighbor_indices[idx1], neighbor_indices[idx2]
                    ]

            if current_dist_sum < min_dist_sum:
                min_dist_sum = current_dist_sum
                best_indices = neighbor_indices

        print("\n4 closest transcriptions to each other:")
        for idx in best_indices:
            print(f"- {filenames[idx]}")
    elif len(embeddings) > 1:
        print(
            f"\nOnly {len(embeddings)} transcriptions available. All are shown in the plot."
        )

    # Show plot if possible (might not work in all CLI environments, but good to have)
    # plt.show()


if __name__ == "__main__":
    main()
