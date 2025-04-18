import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import argparse


def preprocess_titanic_data(skip_scaling: bool = False):
    """Loads, preprocesses, and saves the Titanic dataset.

    Args:
        skip_scaling: If True, bypasses the StandardScaler step.
    """
    # Define relative paths (assuming script is run from workspace root)
    # Or adjust based on where you run the script from
    sprint_dir = "sprints/03_titanic_torch_voyage"
    data_dir = os.path.join(sprint_dir, "data")
    code_dir = os.path.join(sprint_dir, "code")  # Not used yet, but good practice

    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # Adjust output filenames based on scaling flag
    suffix = "_unscaled" if skip_scaling else ""
    processed_train_path = os.path.join(data_dir, f"processed_train{suffix}.csv")
    processed_test_path = os.path.join(data_dir, f"processed_test{suffix}.csv")

    print(f"Loading data from {train_path} and {test_path}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("\n--- Initial Training Data Info ---")
    train_df.info()
    print("\nMissing values in Training Data:")
    print(train_df.isnull().sum())

    print("\n--- Initial Test Data Info ---")
    test_df.info()
    print("\nMissing values in Test Data:")
    print(test_df.isnull().sum())

    # Combine train and test for consistent preprocessing
    # Store PassengerId for later and drop it for now
    train_ids = train_df["PassengerId"]
    test_ids = test_df["PassengerId"]
    all_df = pd.concat([train_df.drop("Survived", axis=1), test_df], ignore_index=True)
    # Keep track of original train/test split size
    n_train = len(train_df)

    # --- Preprocessing Steps Go Here ---
    print("\nStarting preprocessing...")

    # Example: Drop PassengerId and Ticket (often not useful directly)
    print("Dropping PassengerId and Ticket columns...")
    all_df = all_df.drop(["PassengerId", "Ticket"], axis=1)

    # --- Handle Missing Values ---
    print("Handling missing values...")

    # Age: Impute with median
    median_age = all_df["Age"].median()
    print(f"  Imputing missing Age with median: {median_age:.2f}")
    all_df["Age"].fillna(median_age, inplace=True)

    # Fare: Impute with median (only 1 missing value in test set)
    median_fare = all_df["Fare"].median()
    print(f"  Imputing missing Fare with median: {median_fare:.2f}")
    all_df["Fare"].fillna(median_fare, inplace=True)

    # Embarked: Impute with mode (most common port)
    mode_embarked = all_df["Embarked"].mode()[0]
    print(f"  Imputing missing Embarked with mode: {mode_embarked}")
    all_df["Embarked"].fillna(mode_embarked, inplace=True)

    # Cabin: extract deck and create HasCabin feature
    print("  Extracting 'Deck' from 'Cabin' and creating 'HasCabin' feature...")
    all_df["Deck"] = all_df["Cabin"].str[0].fillna("U")  # U for unknown decks
    all_df["HasCabin"] = all_df["Cabin"].notna().astype(int)
    all_df = all_df.drop("Cabin", axis=1)
    print("Missing value handling complete.")

    # --- Feature Engineering ---
    print("\nStarting feature engineering...")

    # Extract Title from Name
    print("  Extracting 'Title' from 'Name'...")
    all_df["Title"] = all_df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
    # Display value counts to see the titles found
    print("    Value counts for extracted titles:")
    print(all_df["Title"].value_counts())

    # Consolidate rare titles
    common_titles = ["Mr", "Miss", "Mrs", "Master"]  # Keep these as they are
    # Titles like 'Dr', 'Rev', 'Col', 'Major', 'Mlle', 'Countess', 'Ms', 'Lady', 'Jonkheer', 'Don', 'Mme', 'Capt', 'Sir' will be replaced
    print("  Consolidating rare titles into 'Other'...")
    all_df["Title"] = all_df["Title"].replace(
        [t for t in all_df["Title"].unique() if t not in common_titles], "Other"
    )
    print("    Value counts after consolidation:")
    print(all_df["Title"].value_counts())

    # Drop the original Name column
    print("  Dropping 'Name' column...")
    all_df = all_df.drop("Name", axis=1)

    # Create FamilySize feature
    print("  Creating 'FamilySize' feature from 'SibSp' and 'Parch'...")
    all_df["FamilySize"] = all_df["SibSp"] + all_df["Parch"] + 1

    # Create IsAlone feature
    print("  Creating 'IsAlone' feature from 'FamilySize'...")
    all_df["IsAlone"] = (all_df["FamilySize"] == 1).astype(int)

    # Drop SibSp and Parch columns
    print("  Dropping 'SibSp' and 'Parch' columns...")
    all_df = all_df.drop(["SibSp", "Parch"], axis=1)

    # Binning Age into categories
    print("  Binning 'Age' into categories...")
    all_df["AgeBin"] = pd.cut(
        all_df["Age"],
        bins=[0, 12, 18, 35, 60, 120],
        labels=["Child", "Teen", "Adult", "MidAge", "Senior"],
    )

    print("Feature engineering complete.")

    # --- Encode Categorical Features ---
    print("\nEncoding categorical features using one-hot encoding...")
    categorical_cols = ["Sex", "Embarked", "Title", "Pclass", "Deck", "AgeBin"]
    print(f"  Columns to encode: {categorical_cols}")
    all_df = pd.get_dummies(
        all_df, columns=categorical_cols, drop_first=False
    )  # drop_first=False is often safer initially
    print(f"  DataFrame shape after encoding: {all_df.shape}")

    # Explicitly convert boolean columns created by get_dummies to int (0/1)
    print("  Converting boolean columns to integers (0/1)...")
    bool_cols = all_df.select_dtypes(include="bool").columns
    all_df[bool_cols] = all_df[bool_cols].astype(int)
    print(f"    Converted columns: {list(bool_cols)}")

    # --- Scale Numerical Features (Conditional) ---
    if not skip_scaling:
        print("\nScaling numerical features using StandardScaler...")
        numerical_cols = [
            "Age",
            "Fare",
            "FamilySize",
        ]  # Assuming HasCabin is already 0/1
        print(f"  Columns to scale: {numerical_cols}")
        scaler = StandardScaler()
        # Fit on the combined data and transform
        all_df[numerical_cols] = scaler.fit_transform(all_df[numerical_cols])
        print("  Numerical features scaled.")
    else:
        print("\nSkipping numerical feature scaling.")

    # --- TODO: Add more preprocessing steps ---
    # - Scaling numerical features (Age, Fare)

    # --- Preprocessing Steps Placeholder ---
    # Add your imputation, feature engineering, encoding here...
    # Remove this section once all steps are implemented
    print(
        "\n--- All planned preprocessing steps implemented --- Removing placeholders ---"
    )

    # Separate back into train and test sets
    print("\nSeparating back into processed train and test sets...")
    processed_train_df = all_df.iloc[:n_train].copy()
    processed_test_df = all_df.iloc[n_train:].copy()

    # Re-add the target variable to the training set
    processed_train_df["Survived"] = train_df["Survived"]

    print("\n--- Processed Training Data Info ---")
    processed_train_df.info()
    print("\nMissing values in Processed Training Data:")
    print(processed_train_df.isnull().sum())

    print("\n--- Processed Test Data Info ---")
    processed_test_df.info()
    print("\nMissing values in Processed Test Data:")
    print(processed_test_df.isnull().sum())

    # Save the processed data
    print(
        f"\nSaving processed data to {processed_train_path} and {processed_test_path}..."
    )
    # Ensure the data directory exists (it should, but belt-and-suspenders)
    os.makedirs(data_dir, exist_ok=True)
    processed_train_df.to_csv(processed_train_path, index=False)
    processed_test_df.to_csv(processed_test_path, index=False)

    print("\nPreprocessing complete and files saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Titanic Data")
    parser.add_argument(
        "--skip_scaling",
        action="store_true",
        help="If set, numerical features (Age, Fare, FamilySize) will not be scaled.",
    )
    args = parser.parse_args()

    preprocess_titanic_data(skip_scaling=args.skip_scaling)
