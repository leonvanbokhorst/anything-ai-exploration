# Sprint 1: The Paws of Dread - Teaching AI Existential Angst via Cat Memes

## Sprint Goal:

To explore fine-tuning a pre-trained language model (from HuggingFace's treasure trove) to generate text that captures the essence of philosophical existential dread, but constrained entirely within the language, style, and common tropes of internet cat memes. Can AI ponder the void if the void is lined with cardboard boxes and dangling strings?

## Tasks / Learning Objectives:

1.  [ ] **Research & Setup:**
    - Identify a suitable pre-trained language model (e.g., GPT-2, DistilGPT2, maybe something smaller/quirkier?) from HuggingFace. **Chosen Model:** `microsoft/Phi-3-mini-4k-instruct`.
    - Set up the PyTorch environment on the chosen machine (recommend the RTX 4090 beast for training, Master!). **Note:** The QLoRA fine-tuning step (Task 3) requires an NVIDIA GPU with CUDA support due to the `bitsandbytes` library. The Windows/WSL2 machine is necessary for this.
    - [x] Install necessary HuggingFace libraries (`transformers`, `datasets`, `peft`, `bitsandbytes`, `accelerate`).
2.  [x] **Data "Collection" (The Fun Part):**
    - [x] Gather or _create_ a small dataset of text snippets embodying "cat meme existentialism." Think "I fits, therefore I sits... but _why_ do I sit?" or "The red dot is meaningless, yet I must chase."
    - [x] Format this dataset appropriately for fine-tuning.
3.  [x] **Fine-Tuning (using QLoRA):**
    - [x] Write and run a PyTorch script using HuggingFace `Trainer` API (or `trl`'s `SFTTrainer`) integrated with `peft` for QLoRA.
    - [x] Configure QLoRA parameters (e.g., rank `r`, `lora_alpha`, target modules).
    - [x] Fine-tune the quantized `Phi-3-mini` model on our bespoke dataset.
    - [x] Experiment with basic hyperparameters (learning rate, epochs). Monitor for signs of comedic despair.
4.  [x] **Generation & Evaluation (The _Really_ Fun Part):**
    - [x] Write a script to generate text from our fine-tuned model.
    - [x] Prompt it with existential queries or cat-related scenarios.
    - [x] Evaluate results based on:
      - Coherence (Does it sound vaguely like... something?)
      - Cat Meme Accuracy (Does it use the lingo? "I can haz meaning?")
      - Existential Angst Quotient (Does it evoke a chuckle of nihilistic recognition?)
5.  [ ] **Documentation & Sharing:**
    - [x] Document the process, findings, and hilarious outputs in `docs/` ([QLoRA Explanation](docs/01_qlora_finetuning_explanation.md)).
    - [x] Save the fine-tuned model (if it's not too embarrassing) in `code/` or `models/` (Adapters pushed to [HF Hub](https://huggingface.co/leonvanbokhorst/phi-3-mini-4k-instruct-existential-cats-v1)).
    - [x] Prepare a few "best of" generated snippets for sharing with unsuspecting non-AI folks. ([See Generated Examples](results/generated_responses.md))

## Definition of Done / Key Questions Answered:

- [x] Environment setup complete and verified.
- [x] A small dataset of "existential cat meme" text exists.
- [x] A language model has been fine-tuned (however successfully) on this data.
- [x] We can generate text from the fine-tuned model.
- [x] Generated examples and key learnings are documented.
- [x] Did the AI achieve enlightenment, or just demand more tuna? (Verdict: Still demands tuna, but ponders *why*.)
- [x] Can absurdity be a valid training signal? (Spoiler: YES!)
