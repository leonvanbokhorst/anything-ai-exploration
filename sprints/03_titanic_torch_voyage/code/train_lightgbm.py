import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import os
import argparse
import time
import optuna
import json

# Global variable to hold data (avoids reloading in each trial)
# Alternatively, use functools.partial if preferred
X_global = None
y_global = None


def objective(trial: optuna.Trial, n_splits: int, random_state: int) -> float:
    """
    Objective function for Optuna hyperparameter optimization.
    Trains and evaluates LightGBM with suggested parameters using K-Fold CV.

    Args:
        trial: An Optuna trial object.
        n_splits: Number of folds for cross-validation.
        random_state: Base random seed.

    Returns:
        The average validation accuracy across the K-Folds.
    """
    global X_global, y_global
    if X_global is None or y_global is None:
        # This should not happen if main is run correctly, but as a safeguard
        raise ValueError("Global data X_global/y_global not loaded.")

    # --- 1. Suggest Hyperparameters ---
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,  # Suppress verbose LightGBM output
        "boosting_type": "gbdt",
        "random_state": random_state,
        "n_jobs": -1,
        # Parameters to be tuned by Optuna
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", 1e-8, 10.0, log=True
        ),  # L1 regularization
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-8, 10.0, log=True
        ),  # L2 regularization
        "subsample": trial.suggest_float(
            "subsample", 0.6, 1.0
        ),  # Fraction of samples for fitting the trees
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.6, 1.0
        ),  # Fraction of features for fitting the trees
        # 'max_depth': trial.suggest_int('max_depth', 3, 10), # Often controlled by num_leaves
        # Consider tuning n_estimators or use a fixed large number with early stopping
    }
    n_estimators = trial.suggest_int("n_estimators", 50, 500)

    # --- 2. K-Fold Cross-Validation ---
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_global, y_global)):
        X_train, X_val = X_global.iloc[train_idx], X_global.iloc[val_idx]
        y_train, y_val = y_global.iloc[train_idx], y_global.iloc[val_idx]

        lgbm = lgb.LGBMClassifier(**params, n_estimators=n_estimators)

        lgbm.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[
                lgb.early_stopping(15, verbose=False)
            ],  # Increased patience slightly
        )

        y_pred = lgbm.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        fold_accuracies.append(accuracy)

        # Optional: Pruning - tell Optuna if a trial is not promising early
        # trial.report(accuracy, fold)
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()

    # --- 3. Return Average Accuracy ---
    avg_accuracy = np.mean(fold_accuracies)
    return avg_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize Titanic LightGBM with Optuna using K-Fold CV"
    )
    sprint_dir = "sprints/03_titanic_torch_voyage"
    default_data_path = os.path.join(sprint_dir, "data", "processed_train_unscaled.csv")

    parser.add_argument(
        "--data_path",
        type=str,
        default=default_data_path,
        help="Path to the processed *unscaled* training data CSV file",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of folds for Stratified K-Fold CV within each trial",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of Optuna optimization trials to run",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # --- Load Data Globally ---
    print(f"Loading data globally from {args.data_path}...")
    try:
        df = pd.read_csv(args.data_path)
        X_global = df.drop("Survived", axis=1)
        y_global = df["Survived"]
        print(f"Loaded {len(df)} samples with {X_global.shape[1]} features globally.")
    except FileNotFoundError:
        print(f"Error: Data file not found at {args.data_path}")
        exit()
    except KeyError:
        print("Error: 'Survived' column not found in the data.")
        exit()

    # --- Run Optuna Study ---
    start_time = time.time()
    # Create a lambda function to pass fixed arguments to the objective
    objective_with_args = lambda trial: objective(
        trial, n_splits=args.n_splits, random_state=args.seed
    )

    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner()
    )
    print(f"\n--- Starting Optuna optimization with {args.n_trials} trials --- ")
    study.optimize(objective_with_args, n_trials=args.n_trials)
    end_time = time.time()

    # --- Print Results ---
    print("\n--- Optuna Optimization Finished ---")
    print(f"Total optimization time: {end_time - start_time:.2f} seconds")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best CV Score (Accuracy): {study.best_value:.6f}")
    print("Best parameters found:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    # Save best parameters to JSON for prediction script
    results_dir = os.path.join(sprint_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    best_params_path = os.path.join(results_dir, "best_params_lgbm.json")
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f)
    print(f"Best parameters saved to: {best_params_path}")

    # --- Optional: Train final model with best params on all data ---
    # print("\nTraining final model with best parameters on all data...")
    # best_params = study.best_params
    # final_model = lgb.LGBMClassifier(
    #     objective='binary',
    #     metric='binary_logloss',
    #     verbosity=-1,
    #     boosting_type='gbdt',
    #     random_state=args.seed,
    #     n_jobs=-1,
    #     **best_params
    # )
    # final_model.fit(X_global, y_global)
    # print("Final model trained.")
    # # Now you could use this final_model to predict on the actual test set
