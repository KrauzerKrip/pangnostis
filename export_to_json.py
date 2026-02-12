import json
from repository import TranscriptionRepository

def main():
    repo = TranscriptionRepository()
    transcriptions = repo.get_all()
    
    if not transcriptions:
        print("No transcriptions found in the database.")
        return

    export_data = []
    for t in transcriptions:
        # Create a dictionary without embedding fields
        data = {
            "id": t.id,
            "filename": t.filename,
            "transcription": t.transcription,
            "who": t.who,
            "what": t.what,
            "where": t.where,
            "context_vector_helper": t.context_vector_helper
        }
        export_data.append(data)

    output_file = "transcriptions_export.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=4, ensure_ascii=False)

    print(f"Successfully exported {len(export_data)} transcriptions to {output_file}")

if __name__ == "__main__":
    main()
