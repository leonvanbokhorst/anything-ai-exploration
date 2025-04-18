# Sprint 3: The Titanic Torch Voyage - Charting Survival with Custom PyTorch

## Sprint Goal:

To design, build, and evaluate a custom PyTorch neural network model for predicting passenger survival on the classic Titanic dataset, exploring feature engineering and network architecture choices.

## Tasks / Learning Objectives:

1.  [ ] **Data Acquisition & Prep:**

    - Download the Titanic dataset (e.g., from Kaggle).
    - Perform exploratory data analysis (EDA) to understand features.
    - Implement robust data preprocessing and feature engineering (handling missing values, encoding categoricals, creating new features). Store processed data in `data/`.

2.  [ ] **Custom Network Design:**

    - Define a custom PyTorch `nn.Module` architecture suitable for tabular data. Consider embedding layers for categorical features, different layer types (linear, activation functions), and regularization (dropout).
    - Implement the model in `code/titanic_model.py`.

3.  [ ] **Training Pipeline:**

    - Implement a PyTorch `Dataset` and `DataLoader` for the Titanic data.
    - Write a training loop (`code/train.py`) including loss function (e.g., BCEWithLogitsLoss), optimizer (e.g., Adam), and evaluation metric (e.g., accuracy).
    - Incorporate logging (e.g., TensorBoard or simple printouts).

4.  [ ] **Train & Evaluate:**

    - Train the custom model on the preprocessed data.
    - Tune hyperparameters (learning rate, network size, dropout rate).
    - Evaluate performance on a validation/test set. Record results in `results/`.
    - Compare against a simple baseline model (e.g., Logistic Regression).

5.  [ ] **Documentation & Reflection:**
    - Document the architecture choices, preprocessing steps, and results in `docs/`.
    - Update this README with findings and link to artifacts.
    - Reflect on the effectiveness of the custom architecture.

## Definition of Done:

- [ ] Titanic data is downloaded, preprocessed, and ready for training.
- [ ] Custom PyTorch model (`titanic_model.py`) is implemented.
- [ ] Training script (`train.py`) runs and trains the model.
- [ ] Model performance is evaluated and compared to a baseline.
- [ ] Key findings and artifacts are documented.
