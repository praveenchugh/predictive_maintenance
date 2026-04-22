
"""
Data Preparation Module
-----------------------
Loads dataset from Hugging Face, performs train/test split,
and uploads prepared splits back to the dataset repository.

Authentication:
- Uses environment variable `HF_TOKEN`
"""

# ============================================================
# Imports
# ============================================================
import os
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from huggingface_hub import HfApi


# ============================================================
# Configuration
# ============================================================
DATASET_REPO = "praveenchugh/capstone-predictive-maintenance-dataset"
DATASET_FILENAME = "engine_data.csv"
DATASET_PATH = f"hf://datasets/{DATASET_REPO}/{DATASET_FILENAME}"

TARGET_COL = "Engine Condition"
TEST_SIZE = 0.2
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "Engine rpm",         
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",     
    "Coolant temp",
    "Engine Condition",
]

CATEGORICAL_FEATURES = [
    "Engine Condition",
]

SPLIT_FILES = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]


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
# Data Loading
# ============================================================
def load_dataset() -> pd.DataFrame:
    """
    Load dataset directly from Hugging Face storage.
    """
    df = pd.read_csv(DATASET_PATH)
    print("Dataset loaded successfully from Hugging Face.")
    return df


# ============================================================
# Data Preparation
# ============================================================
def prepare_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataset into training and testing sets and save locally.
    """
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # ensures class balance in splits
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print("Train/test split completed and saved locally.")

    return Xtrain, Xtest, ytrain.squeeze(), ytest.squeeze()


# ============================================================
# Upload Prepared Splits
# ============================================================
def upload_dataset_splits(api: HfApi) -> None:
    """
    Upload prepared dataset split files to Hugging Face dataset repository.
    """
    for file in SPLIT_FILES:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Missing split file: {file}")

        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=file,
            repo_id=DATASET_REPO,
            repo_type="dataset",
        )

    print("Dataset splits uploaded successfully to Hugging Face.")


# ============================================================
# Main Execution
# ============================================================
def main() -> None:
    """
    End-to-end data preparation workflow.
    """
    api = get_hf_client()

    df = load_dataset()
    prepare_data(df)
    upload_dataset_splits(api)

    print("Data preparation pipeline completed.")


if __name__ == "__main__":
    main()
