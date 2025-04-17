# Project Progress Log

_(This log will track major milestones, completed sprints, and key learnings across our journey into the wonderfully weird world of AI.)_

---

## Sprint 1: The Paws of Dread (Completed)

*   **Goal:** Fine-tune Phi-3-mini with QLoRA to generate text blending cat memes and existential dread.
*   **Outcome:** Successfully fine-tuned the model! It generates hilariously angsty cat-themed text, although requires careful prompting/generation settings to terminate properly.
*   **Key Learnings:**
    *   QLoRA significantly reduces memory requirements for fine-tuning.
    *   `trl`'s `SFTTrainer` simplifies the process, but API changes require attention to documentation versions.
    *   Explicitly adding EOS tokens to training data is crucial for reliable generation termination.
    *   Pinning dependency versions (`~=`) in `pyproject.toml` is vital for reproducibility and avoiding subtle library mismatches.
*   **Artifacts:**
    *   [Sprint README](sprints/01_paws_of_dread/README.md)
    *   [Fine-tuned Adapters (HF Hub)](https://huggingface.co/leonvanbokhorst/phi-3-mini-4k-instruct-existential-cats-v1)
    *   [Generated Examples](sprints/01_paws_of_dread/results/generated_responses.md)
    *   [Training Script](sprints/01_paws_of_dread/code/train_qlora_phi3_cats.py)
    *   [Generation Script](sprints/01_paws_of_dread/code/generate_text.py)

---
