import numpy as np
from sklearn.ensemble import IsolationForest
from repository import TranscriptionRepository

def analyze_anomalies(embeddings, filenames, emb_type):
    """Perform anomaly detection on a specific set of embeddings."""
    print(f"\n--- Analyzing '{emb_type}' embeddings for anomalies ---")
    
    # Isolation Forest
    # contamination: expected proportion of outliers in the data
    contamination = min(0.1, 2.0 / len(embeddings)) if len(embeddings) > 10 else 0.1
    
    clf = IsolationForest(contamination=contamination, random_state=42)
    preds = clf.fit_predict(embeddings)
    
    # -1 for outliers, 1 for inliers
    anomaly_indices = np.where(preds == -1)[0]
    
    if len(anomaly_indices) == 0:
        print(f"No anomalies found in '{emb_type}' data.")
    else:
        print(f"Found {len(anomaly_indices)} potential anomalies in '{emb_type}':")
        scores = clf.decision_function(embeddings)
        for idx in anomaly_indices:
            print(f"- {filenames[idx]} (Score: {scores[idx]:.4f})")

def main():
    repo = TranscriptionRepository()
    transcriptions = repo.get_all()
    
    if not transcriptions:
        print("No transcriptions found in the database.")
        return
    
    if len(transcriptions) < 5:
        print(f"Only {len(transcriptions)} transcriptions found. Need at least 5 for meaningful anomaly detection.")
        return

    # Define embedding types and their corresponding attribute names in the Transcription model
    embedding_map = {
        "who": "who_embedding",
        "what": "what_embedding",
        "when": "when_embedding",
        "where": "where_embedding",
        "context": "context_embedding"
    }

    filenames = [t.filename for t in transcriptions]
    
    for emb_type, attr_name in embedding_map.items():
        embeddings = np.array([getattr(t, attr_name) for t in transcriptions])
        analyze_anomalies(embeddings, filenames, emb_type)

if __name__ == "__main__":
    main()