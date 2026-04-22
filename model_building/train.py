
# ============================================================
# END-TO-END ML PIPELINE WITH HUGGING FACE + GBM TUNING
# ============================================================

import datasets
import json
import joblib

from huggingface_hub import HfApi, upload_file
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# LOAD DATA
# ============================================================

dataset = datasets.load_dataset(
    "praveenchugh/capstone-predictive-maintenance-dataset"
)

df = dataset["train"].to_pandas()

# Reduce memory usage
for col in df.select_dtypes(include=["float64"]).columns:
    df[col] = df[col].astype("float32")


# ============================================================
# PREPARE DATA
# ============================================================

# Detect all target-like columns
target_cols = [c for c in df.columns if "Engine Condition" in c]

# Keep only ONE as target
target_column = target_cols[0]

# Drop ALL target-related columns from features
X = df.drop(columns=target_cols)
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# PIPELINE
# ============================================================

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", HistGradientBoostingClassifier(random_state=42))
])


# ============================================================
# HYPERPARAMETER TUNING
# ============================================================

param_dist = {
    "model__max_iter": [100, 200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [3, 5, None],
    "model__min_samples_leaf": [20, 50],
    "model__l2_regularization": [0.0, 0.1]
}

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=10,
    cv=2,
    scoring="accuracy",
    n_jobs=2,
    verbose=1,
    random_state=42
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("Training model...")
search.fit(X_train, y_train)

best_model = search.best_estimator_
best_params = search.best_params_

print("Best parameters found:", best_params)


# ============================================================
# SAVE PARAMETERS
# ============================================================

with open("best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)


# ============================================================
# EVALUATION
# ============================================================

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model accuracy: {accuracy}")
print(report)

with open("evaluation.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n\n")
    f.write(report)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(best_model, "gbm_model.joblib", compress=3)


# ============================================================
# UPLOAD TO HUGGING FACE
# ============================================================

repo_id = "praveenchugh/engine-condition-gbm-model"

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

upload_file(
    path_or_fileobj="gbm_model.joblib",
    path_in_repo="gbm_model.joblib",
    repo_id=repo_id,
    repo_type="model"
)

upload_file(
    path_or_fileobj="best_params.json",
    path_in_repo="best_params.json",
    repo_id=repo_id,
    repo_type="model"
)

upload_file(
    path_or_fileobj="evaluation.txt",
    path_in_repo="evaluation.txt",
    repo_id=repo_id,
    repo_type="model"
)


# ============================================================
# COMPLETION
# ============================================================

print("Model trained and uploaded successfully")
