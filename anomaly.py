import numpy as np
from sklearn.ensemble import IsolationForest
from repository import TranscriptionRepository

def main():
    repo = TranscriptionRepository()
    transcriptions = repo.get_all()
    
    if not transcriptions:
        print("No transcriptions found in the database.")
        return
    
    if len(transcriptions) < 5:
        print(f"Only {len(transcriptions)} transcriptions found. Need at least 5 for meaningful anomaly detection.")
        return

    # Extract embeddings and filenames
    embeddings = np.array([t.embedding for t in transcriptions])
    filenames = [t.filename for t in transcriptions]
    
    print(f"Analyzing {len(embeddings)} embeddings for anomalies using Isolation Forest...")
    
    # Isolation Forest
    # contamination: expected proportion of outliers in the data
    # We'll set it to 0.1 (10%) but adjust if there are very few samples
    contamination = min(0.1, 2.0 / len(embeddings)) if len(embeddings) > 10 else 0.1
    
    clf = IsolationForest(contamination=contamination, random_state=42)
    preds = clf.fit_predict(embeddings)
    
    # -1 for outliers, 1 for inliers
    anomaly_indices = np.where(preds == -1)[0]
    
    if len(anomaly_indices) == 0:
        print("No anomalies found based on the current model.")
    else:
        print(f"\nFound {len(anomaly_indices)} potential anomalies:")
        for idx in anomaly_indices:
            # We can also get the anomaly score
            scores = clf.decision_function(embeddings)
            print(f"- {filenames[idx]} (Score: {scores[idx]:.4f})")
            
        print("\nNote: Lower (more negative) scores indicate more anomalous data points.")

if __name__ == "__main__":
    main()
