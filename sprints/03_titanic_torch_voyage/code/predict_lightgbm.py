import pandas as pd
import lightgbm as lgb
import os
import json


def main():
    sprint_dir = "sprints/03_titanic_torch_voyage"
    data_dir = os.path.join(sprint_dir, "data")
    results_dir = os.path.join(sprint_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Load best hyperparameters from JSON
    best_params_path = os.path.join(results_dir, "best_params_lgbm.json")
    print(f"Loading best LightGBM params from {best_params_path}...")
    with open(best_params_path, "r") as f:
        best_params = json.load(f)
    print(f"Best parameters: {best_params}")

    train_path = os.path.join(data_dir, "processed_train_unscaled.csv")
    test_path = os.path.join(data_dir, "processed_test_unscaled.csv")
    orig_test_path = os.path.join(data_dir, "test.csv")

    print(f"Loading training data from {train_path} and test data from {test_path}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    orig_test_df = pd.read_csv(orig_test_path)

    # Separate features and target
    X_train = train_df.drop("Survived", axis=1)
    y_train = train_df["Survived"]
    X_test = test_df

    # Instantiate model using loaded params
    print("Initializing LightGBM with loaded parameters...")
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        verbosity=-1,
        boosting_type="gbdt",
        random_state=42,
        n_jobs=-1,
        **best_params,
    )

    # Train on full data
    print("Training final LightGBM model on full training data...")
    model.fit(X_train, y_train)

    # Predict on test set
    print("Generating predictions for test set...")
    preds = model.predict(X_test).astype(int)

    # Create submission DataFrame
    submission = pd.DataFrame(
        {"PassengerId": orig_test_df["PassengerId"], "Survived": preds}
    )

    # Save to CSV
    submission_path = os.path.join(results_dir, "lightgbm_submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission file saved to: {submission_path}")


if __name__ == "__main__":
    main()
