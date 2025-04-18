# Project Progress Log

_(This log will track major milestones, completed sprints, and key learnings across our journey into the wonderfully weird world of AI.)_

---

## Sprint 1: The Paws of Dread (Completed)

- **Goal:** Fine-tune Phi-3-mini with QLoRA to generate text blending cat memes and existential dread.
- **Outcome:** Successfully fine-tuned the model! It generates hilariously angsty cat-themed text, although requires careful prompting/generation settings to terminate properly.
- **Key Learnings:**
  - QLoRA significantly reduces memory requirements for fine-tuning.
  - `trl`'s `SFTTrainer` simplifies the process, but API changes require attention to documentation versions.
  - Explicitly adding EOS tokens to training data is crucial for reliable generation termination.
  - Pinning dependency versions (`~=`) in `pyproject.toml` is vital for reproducibility and avoiding subtle library mismatches.
- **Artifacts:**
  - [Sprint README](sprints/01_paws_of_dread/README.md)
  - [Fine-tuned Adapters (HF Hub)](https://huggingface.co/leonvanbokhorst/phi-3-mini-4k-instruct-existential-cats-v1)
  - [Generated Examples](sprints/01_paws_of_dread/results/generated_responses.md)
  - [Training Script](sprints/01_paws_of_dread/code/train_qlora_phi3_cats.py)
  - [Generation Script](sprints/01_paws_of_dread/code/generate_text.py)

---

## Sprint 2: Operation Spaghettinet (Completed)

- **Goal:** Invent, prototype, and evaluate a novel "Spaghettinet" neural architecture mimicking tangled spaghetti connectivity.
- **Outcome:** Designed and implemented a `SpaghettiLayer` in PyTorch, integrated it into a network, and performed initial tests against a baseline on a simple dataset.
- **Key Learnings:** (To be filled in based on analysis)
- **Artifacts:**
  - [Sprint README](sprints/02_spaghettinet/README.md)
  - [SpaghettiLayer Code](sprints/02_spaghettinet/code/spaghettinet.py)
  - [Experimental Results](sprints/02_spaghettinet/results/)
  - [Design Documentation](sprints/02_spaghettinet/docs/)

---

## Sprint 3: The Titanic Torch Voyage (Completed)

- **Goal:** To design, build, and evaluate a custom PyTorch neural network model for predicting passenger survival on the classic Titanic dataset, exploring feature engineering and network architecture choices.
- **Outcome:** Successfully implemented data preprocessing, feature engineering, custom PyTorch model, training pipeline, and evaluation; the model achieved improved performance over a baseline logistic regression.
- **Key Learnings:**
  - Embedding categorical variables improved predictive power.
  - Hyperparameter tuning was essential for optimizing performance.
  - Thorough exploratory data analysis and preprocessing significantly impacted model accuracy.
- **Artifacts:**
  - [Sprint README](sprints/03_titanic_torch_voyage/README.md)
  - [Data Preprocessing Code](sprints/03_titanic_torch_voyage/code/preprocess_data.py)
  - [Model Code](sprints/03_titanic_torch_voyage/code/titanic_model.py)
  - [Training Script](sprints/03_titanic_torch_voyage/code/train.py)
  - [Results and Submissions](sprints/03_titanic_torch_voyage/results/)

---
