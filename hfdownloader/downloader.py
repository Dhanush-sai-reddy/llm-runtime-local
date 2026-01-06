import os
import argparse
from huggingface_hub import snapshot_download, login

def download_model(repo_id, token=None):
    base_dir = "/data" 
    folder_name = repo_id.replace("/", "--")
    local_dir = os.path.join(base_dir, folder_name)

    print(f"Downloading: {repo_id}")
    print(f"Destination: {local_dir}")

    try:
        if token:
            login(token=token)
            print(" Authenticated")

        path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False, # Downloads actual files, not links
            resume_download=True
        )
        print(f"Finished. Saved to: {path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", type=str)
    parser.add_argument("--token", type=str, default=os.getenv("HF_TOKEN"))
    args = parser.parse_args()
    
    download_model(args.model_id, args.token)