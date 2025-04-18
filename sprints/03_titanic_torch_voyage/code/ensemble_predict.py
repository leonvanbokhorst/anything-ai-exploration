"""
ensemble_predict.py: Ensemble LightGBM and PyTorch TitanicNet predictions.
"""

import os
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
from torch.utils.data import DataLoader
from titanic_model import TitanicNet
from titanic_dataset import TitanicDataset


def main():
    sprint_dir = "sprints/03_titanic_torch_voyage"
    data_dir = os.path.join(sprint_dir, "data")
    results_dir = os.path.join(sprint_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # File paths for LightGBM (unscaled with new features)
    train_csv = os.path.join(data_dir, "processed_train_unscaled.csv")
    test_csv = os.path.join(data_dir, "processed_test_unscaled.csv")
    original_test_csv = os.path.join(data_dir, "test.csv")
    params_path = os.path.join(results_dir, "best_params_lgbm.json")
    pt_model_path = os.path.join(results_dir, "titanic_model.pth")
    # Use scaled processed_test.csv for PyTorch (to match training data scaling)
    pt_test_csv = os.path.join(data_dir, "processed_test.csv")

    # Load LightGBM hyperparameters
    print(f"Loading LightGBM hyperparameters from {params_path}...")
    with open(params_path, "r") as f:
        lgb_params = json.load(f)

    # Load data for LightGBM
    print("Loading train/test data for LightGBM...")
    df_train = pd.read_csv(train_csv)
    X_train = df_train.drop("Survived", axis=1)
    y_train = df_train["Survived"]
    df_test = pd.read_csv(test_csv)
    X_test = df_test
    original_test = pd.read_csv(original_test_csv)

    # Train LightGBM model
    print("Training LightGBM model on full training data...")
    lgb_model = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        verbosity=-1,
        boosting_type="gbdt",
        random_state=42,
        n_jobs=-1,
        **lgb_params,
    )
    lgb_model.fit(X_train, y_train)
    lgb_probs = lgb_model.predict_proba(X_test)[:, 1]

    # Load and predict with PyTorch model using old processed features
    print("Loading TitanicDataset and PyTorch model for test data...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = TitanicDataset(pt_test_csv)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    pt_model = TitanicNet(input_features=test_dataset.n_features)
    print(f"Loading PyTorch model weights from {pt_model_path}...")
    pt_model.load_state_dict(torch.load(pt_model_path, map_location=device))
    pt_model.to(device)
    pt_model.eval()

    pt_probs = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch.to(device)
            outputs = pt_model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            pt_probs.extend(probs.tolist())
    pt_probs = np.array(pt_probs)

    # Ensemble averaging
    print("Averaging probabilities from LightGBM and PyTorch models...")
    ensemble_probs = (lgb_probs + pt_probs) / 2
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)

    # Save ensemble submission
    submission = pd.DataFrame(
        {
            "PassengerId": original_test["PassengerId"],
            "Survived": ensemble_preds,
        }
    )
    submission_path = os.path.join(results_dir, "ensemble_submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Ensemble submission saved to {submission_path}")


if __name__ == "__main__":
    main()
