# Explanation: QLoRA Fine-tuning Script 

Code: [train_qlora_phi3_cats.py](../code/train_qlora_phi3_cats.py)

This document explains the Python script `sprints/01_paws_of_dread/code/train_qlora_phi3_cats.py`, which is designed to fine-tune the `microsoft/Phi-3-mini-4k-instruct` language model on our custom "existential cat memes" dataset using the QLoRA technique.

## Purpose

The goal is to adapt the pre-trained Phi-3 model to generate text that mimics the style and content of internet cat memes while simultaneously expressing themes of philosophical existential dread. We want to see if we can teach the AI to ponder the void, but only using vocabulary like "I can haz meaning?" or "The red dot is futile."

## Key Components & Technologies

1.  **Base Model:** `microsoft/Phi-3-mini-4k-instruct`
    *   A relatively small, powerful instruction-tuned model chosen as the starting point.
2.  **Dataset:** `leonvanbokhorst/existential-cat-memes-v1` (on Hugging Face Hub)
    *   Our custom-curated dataset containing text snippets embodying feline angst. Loaded directly from the Hub.
3.  **Quantization (BitsAndBytes):**
    *   The script uses the `BitsAndBytesConfig` to load the base model in 4-bit precision (`load_in_4bit=True`).
    *   Specifically, it uses NF4 (`bnb_4bit_quant_type="nf4"`) quantization with `bfloat16` compute data type (`bnb_4bit_compute_dtype=torch.bfloat16`) and double quantization (`bnb_4bit_use_double_quant=True`).
    *   **Why?** This significantly reduces the model's memory footprint (GPU VRAM), making it feasible to fine-tune on consumer hardware like the RTX 4090.
4.  **QLoRA (PEFT):**
    *   Quantization-Aware Low-Rank Adaptation. Instead of fine-tuning the entire quantized model (which is difficult), QLoRA freezes the 4-bit base model weights and injects small, trainable "adapter" layers (LoRA layers) into specific modules (configured in `lora_target_modules`).
    *   The `LoraConfig` specifies the rank (`lora_r`), scaling (`lora_alpha`), dropout, and target modules for these adapters.
    *   **Why?** This allows for very memory-efficient fine-tuning, as only the adapter weights (a tiny fraction of the total parameters) are updated.
    *   **Novice Explanation:** Imagine a giant, complex machine (the base model) that already knows a lot. Training the whole machine would require a huge factory (lots of GPU memory). QLoRA is like adding a few small, specialized tools (LoRA adapters) to the machine. We freeze the big machine (keep its original knowledge) and only train these new tools to learn our specific task (generating existential cat memes). Because the tools are small, the training factory needed is much smaller and cheaper (less VRAM). Quantization (the 'Q' in QLoRA) is like making the big machine itself take up less space *before* we even add the tools, further saving resources.
5.  **Trainer (`trl.SFTTrainer`):**
    *   The `SFTTrainer` (Supervised Fine-tuning Trainer) from the `trl` library simplifies the training loop.
    *   It handles tokenization, formatting the dataset (taking the `text` field), applying the PEFT configuration (LoRA adapters), and managing the training process based on `TrainingArguments`.
6.  **`TrainingArguments`:**
    *   This class holds all hyperparameters and settings for the training run, such as:
        *   `output_dir`: Where to save checkpoints locally.
        *   `num_train_epochs`, `per_device_train_batch_size`, `learning_rate`: Standard training parameters.
        *   `optim="paged_adamw_32bit"`: Memory-efficient optimizer suitable for QLoRA.
        *   `bf16=True`: Enables mixed-precision training using bfloat16 (requires compatible GPU).
        *   `gradient_checkpointing=True`: Further saves memory by recomputing activations during the backward pass instead of storing them.
        *   `push_to_hub=True`, `hub_model_id`: Automatically pushes the trained LoRA adapters to the specified Hugging Face Hub repository (`leonvanbokhorst/phi-3-mini-4k-instruct-existential-cats-v1`) upon completion or at save points.

## Hyperparameter Rationale (Why these settings?)

Many hyperparameters involve trade-offs between performance, training time, and memory usage. The values in the script are common starting points, but may need tuning:

*   **`lora_r = 16` (Rank):** Think of this as the 'complexity' or 'capacity' of the LoRA adapters. Higher ranks can capture more complex patterns but use more memory and can sometimes overfit. Lower ranks are more parameter-efficient. `16` or `32` are common starting points balancing performance and efficiency.
*   **`lora_alpha = 32` (Alpha):** This is a scaling factor for the LoRA weights, often set to double the rank (`2 * lora_r`). It helps balance the influence of the original model weights and the new adapter weights. The `r`/`alpha` ratio matters.
*   **`lora_dropout = 0.05`:** Standard dropout applied to the LoRA layers to prevent overfitting (the adapters becoming too specialized to the training data).
*   **`lora_target_modules = [...]`:** Specifies *which* parts of the base model get the LoRA adapters. Targeting attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and feed-forward layers (`gate_proj`, `up_proj`, `down_proj`) is standard practice for transformer models like Phi-3, as these layers are crucial for learning patterns.
*   **`num_train_epochs = 3`:** How many times the model sees the entire dataset. Too few epochs, and the model might not learn enough; too many, and it might overfit or training takes too long. 1-3 epochs is often sufficient for fine-tuning, especially with small datasets.
*   **`per_device_train_batch_size = 2`:** How many examples are processed simultaneously on one GPU. Limited by VRAM. Smaller batches use less memory but can lead to noisier gradient updates (requiring potentially smaller learning rates). `2` is a conservative value suitable for larger models or limited VRAM.
*   **`gradient_accumulation_steps = 1`:** Allows simulating a larger effective batch size without using more memory. If set to `4`, gradients are computed for 4 small batches (`per_device_train_batch_size`) before updating the model weights, simulating `batch_size * accumulation_steps`. We start with `1` for simplicity.
*   **`optim = "paged_adamw_32bit"`:** AdamW is a standard optimizer. The "paged" version is crucial for QLoRA as it offloads optimizer states to CPU RAM, significantly reducing VRAM usage during training.
*   **`learning_rate = 2e-4`:** How big the update steps are for the LoRA weights. Needs careful tuning. Too high, and training might become unstable; too low, and it might take too long or get stuck. `1e-4` to `3e-4` is a typical range for LoRA fine-tuning.
*   **`bf16 = True`:** Use BFloat16 mixed precision. This speeds up computation and reduces memory slightly compared to FP32, especially on modern GPUs that support it well (like the RTX 4090). It requires `bnb_4bit_compute_dtype=torch.bfloat16` in the `BitsAndBytesConfig`.
*   **`gradient_checkpointing = True`:** A memory-saving technique where activations are recalculated during the backward pass instead of being stored. Slows down training slightly but can drastically reduce VRAM usage, often necessary for larger models or sequence lengths.

## Workflow

1.  **Configuration:** Set model IDs, QLoRA parameters, quantization settings, and training arguments.
2.  **Load Components:** Load the tokenizer, the 4-bit quantized base model, and the dataset from the Hub.
3.  **Prepare Model:** Prepare the quantized model for training using `prepare_model_for_kbit_training`.
4.  **Setup Trainer:** Initialize `SFTTrainer` with the model, dataset, tokenizer, LoRA config, and training arguments.
5.  **Train:** Call `trainer.train()` to start the fine-tuning process. The trainer handles gradient updates only for the LoRA adapter weights.
6.  **Save & Push:** After training, save the trained LoRA adapters locally (`trainer.save_model()`) and push them to the Hugging Face Hub (handled by `TrainingArguments`).

## Expected Outcome

The script will produce:
*   Training logs and checkpoints saved locally in the `results-existential-cats` directory.
*   The final trained LoRA adapter weights uploaded to the Hugging Face Hub repository `leonvanbokhorst/phi-3-mini-4k-instruct-existential-cats-v1`. These adapters can then be loaded on top of the original `Phi-3-mini` model for inference.

The resulting model should, hopefully, generate text reflecting the unique blend of cat meme language and existential dread present in our training data.

## Note on FlashAttention Warnings

You might encounter warnings related to `flash-attention` (e.g., "'flash-attention' package not found") when running training scripts or loading models.

*   **What it is:** FlashAttention is an optional, highly optimized implementation of the attention mechanism used in transformer models. It can significantly speed up training and reduce memory usage on compatible NVIDIA GPUs.
*   **Why the warning?** The warning appears because the necessary `flash-attn` library is not installed in the environment.
*   **Is it required?** No. The training script `train_qlora_phi3_cats.py` currently has the line to enable FlashAttention (`attn_implementation="flash_attention_2"`) commented out. It will default to the standard PyTorch attention mechanism (`"eager"`), which works correctly but might be slower.
*   **Action:** For now, these warnings can be safely ignored. If performance becomes an issue later, we could investigate installing `flash-attn` and enabling it in the script. 