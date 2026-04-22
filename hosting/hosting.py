"""
Hugging Face Space Deployment Utility
-------------------------------------
Uploads the local deployment folder (Streamlit + Docker setup)
to a Hugging Face Space.

Authentication:
- Uses HF token from environment variable `HF_TOKEN`.
"""

# ============================================================
# Imports
# ============================================================
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError


# ============================================================
# Configuration
# ============================================================
HF_TOKEN_ENV = "HF_TOKEN"

LOCAL_DEPLOYMENT_FOLDER = "deployment"

SPACE_REPO_ID = "praveenchugh/engine-condition-app"
SPACE_SUBFOLDER_PATH = ""   # keep empty unless you want subdirectory


# ============================================================
# Authentication
# ============================================================
def get_hf_client() -> HfApi:
    """
    Create authenticated Hugging Face client using HF token.
    """
    token = os.getenv(HF_TOKEN_ENV)

    if not token:
        raise RuntimeError(
            "HF_TOKEN not found. Set it using environment variables."
        )

    return HfApi(token=token)


# ============================================================
# Repository utilities
# ============================================================
def ensure_space_exists(api: HfApi, repo_id: str) -> None:
    """
    Ensure Hugging Face Space exists.
    If not present, create a new Space using Docker SDK.
    """
    try:
        api.repo_info(repo_id=repo_id, repo_type="space")
        print(f"Space already exists: {repo_id}")

    except RepositoryNotFoundError:
        print(f"Creating new Space: {repo_id}")

        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",   # REQUIRED (streamlit no longer valid)
            private=False
        )

        print("Space created successfully.")


# ============================================================
# Deployment
# ============================================================
def upload_space(api: HfApi) -> None:
    """
    Upload local deployment folder to Hugging Face Space.
    """
    if not os.path.exists(LOCAL_DEPLOYMENT_FOLDER):
        raise FileNotFoundError(
            f"Deployment folder not found: {LOCAL_DEPLOYMENT_FOLDER}"
        )

    print(f"Uploading files to Space: {SPACE_REPO_ID}")

    api.upload_folder(
        folder_path=LOCAL_DEPLOYMENT_FOLDER,
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        path_in_repo=SPACE_SUBFOLDER_PATH
    )

    print("Upload completed successfully.")


# ============================================================
# Main execution
# ============================================================
def main() -> None:
    """
    End-to-end deployment:
    1. Authenticate
    2. Ensure Space exists
    3. Upload files
    """
    api = get_hf_client()
    ensure_space_exists(api, SPACE_REPO_ID)
    upload_space(api)


if __name__ == "__main__":
    main()
