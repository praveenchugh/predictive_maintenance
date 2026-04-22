
"""
Hugging Face Dataset Registration
--------------------------------
Uploads the local predictive maintenance dataset to Hugging Face Hub.

Authentication:
- Uses environment variable `HF_TOKEN`
"""

# ============================================================
# Imports
# ============================================================
import os
from typing import Optional

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError


# ============================================================
# Constants
# ============================================================
REPO_ID = "praveenchugh/capstone-predictive-maintenance-dataset"
REPO_TYPE = "dataset"
DATA_PATH = "capstone_predictive_maintenance/data"


# ============================================================
# Authentication
# ============================================================
def get_hf_token() -> str:
    """
    Retrieve Hugging Face token from environment variable.

    Raises
    ------
    RuntimeError
        If HF_TOKEN is not set.
    """
    token: Optional[str] = os.getenv("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable not set. "
            "Configure it locally or in GitHub Actions secrets."
        )

    return token


def get_hf_client() -> HfApi:
    """
    Create an authenticated Hugging Face API client.
    """
    return HfApi(token=get_hf_token())


# ============================================================
# Repository Utilities
# ============================================================
def ensure_repo_exists(api: HfApi, repo_id: str, repo_type: str) -> None:
    """
    Ensure the Hugging Face repository exists.
    Creates it if missing.
    """
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Repository already exists: {repo_id}")

    except RepositoryNotFoundError:
        print(f"Repository not found. Creating: {repo_id}")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print("Repository created successfully.")


# ============================================================
# Dataset Upload
# ============================================================
def upload_dataset(api: HfApi, repo_id: str, data_path: str) -> None:
    """
    Upload local dataset folder to Hugging Face Hub.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    print(f"Uploading dataset from '{data_path}' to '{repo_id}'...")

    api.upload_folder(
        folder_path=data_path,
        repo_id=repo_id,
        repo_type="dataset",
    )

    print("Dataset upload completed successfully.")


# ============================================================
# Main Execution
# ============================================================
def main() -> None:
    """
    End-to-end dataset registration workflow.
    """
    api = get_hf_client()

    ensure_repo_exists(api, REPO_ID, REPO_TYPE)
    upload_dataset(api, REPO_ID, DATA_PATH)

    print("Dataset registration finished.")


if __name__ == "__main__":
    main()
