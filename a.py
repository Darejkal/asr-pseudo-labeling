import os
import json
from huggingface_hub import HfApi, create_repo, delete_repo
# Configuration
REPO_ID = "Darejkal/automatic_extraction_data_20250115"
LOCAL_DATA_PATH = "./results"  # Change this to your folder path
TEMP_UPLOAD_DIR = "./results_filtered"

api = HfApi()

def clean_and_upload():
    # 1. Delete all existing data from the repo
    print(f"Deleting existing data from {REPO_ID}...")
    try:
        print(f"Resetting repository: {REPO_ID}")
        delete_repo(repo_id=REPO_ID, repo_type="dataset")
        create_repo(repo_id=REPO_ID, repo_type="dataset", private=False) # Set private=False if public
    except Exception as e:
        print(f"Note: Repo did not exist or could not be reset: {e}")

    # 2. Filter local JSON files
    if not os.path.exists(TEMP_UPLOAD_DIR):
        os.makedirs(TEMP_UPLOAD_DIR)

    print("Filtering files...")
    for filename in os.listdir(LOCAL_DATA_PATH):
        if filename.endswith(".json"):
            file_path = os.path.join(LOCAL_DATA_PATH, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Logic: Keep if "language" key is MISSING in the first level dict
                if "language" not in data:
                    # Save to temp directory for clean upload
                    with open(os.path.join(TEMP_UPLOAD_DIR, filename), 'w') as f:
                        json.dump(data, f, indent=4,ensure_ascii=False)
                else:
                    print(filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # 3. Upload the remaining files
    print(f"Uploading filtered data to {REPO_ID}...")
    api.upload_folder(
        folder_path=TEMP_UPLOAD_DIR,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Filtered out entries containing 'language' key"
    )
    print("Process complete.")

if __name__ == "__main__":
    clean_and_upload()