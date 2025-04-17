# QLoRA Fine-tuning script for Phi-3-mini on Existential Cat Memes

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer
import os # For reading potential API key from env
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
# Model IDs
hf_username = os.getenv("HF_USERNAME")
base_model_id = "microsoft/Phi-3-mini-4k-instruct"
dataset_id = f"{hf_username}/existential-cat-memes-v1" # The one we just uploaded!
new_model_id = f"{hf_username}/phi-3-mini-4k-instruct-existential-cats-v1" # Where to save the fine-tuned model

# QLoRA config
lora_r = 16                # Rank
lora_alpha = 32            # Alpha scaling factor
lora_dropout = 0.05        # Dropout probability
lora_target_modules = [    # Modules to apply LoRA to (specific to Phi-3 - may need adjustment)
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# BitsAndBytes config (4-bit quantization)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",             # NF4 quantization type
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute dtype for faster training
    bnb_4bit_use_double_quant=True,        # Use double quantization
)

# Training arguments
output_dir = "./results-existential-cats" # Local directory to save checkpoints
num_train_epochs = 3                      # Number of training epochs (Reverted back from 5)
per_device_train_batch_size = 2           # Batch size per GPU
gradient_accumulation_steps = 1           # Accumulate gradients over steps
optim = "paged_adamw_32bit"               # Paged AdamW optimizer for memory efficiency
save_steps = 50                           # Save checkpoint every N steps
logging_steps = 10                        # Log training info every N steps
learning_rate = 2e-4                      # Learning rate
max_grad_norm = 0.3                       # Gradient clipping max norm
max_steps = -1                            # Max training steps (if not using epochs)
warmup_ratio = 0.03                       # Warmup ratio for learning rate scheduler
lr_scheduler_type = "constant"            # Learning rate scheduler type ("cosine" is also common)
gradient_checkpointing = True             # Use gradient checkpointing to save memory

# --- Hugging Face Hub Authentication (Optional - needed for pushing model) ---
# It should use the cached token from `huggingface-cli login`
# If you need to pass a token explicitly (e.g., in environments without login),
# you might uncomment the following line and ensure HF_TOKEN is set:
# from huggingface_hub import login
# login(token=os.environ.get("HF_TOKEN"))

# --- Load Tokenizer ---
print(f"Loading tokenizer for {base_model_id}...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
# Set padding side and pad token (common practice for decoder-only models)
tokenizer.padding_side = 'right'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # Use EOS token if no pad token exists

# --- Load Base Model with Quantization ---
print(f"Loading base model {base_model_id} with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto", # Automatically map layers to available devices (GPU/CPU)
    trust_remote_code=True,
    # attn_implementation="flash_attention_2" # Optional: Requires flash-attn library
)
model.config.use_cache = False # Disable caching for training

# --- Prepare Model for QLoRA ---
print("Preparing model for K-bit training (QLoRA)...")
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)

# --- LoRA Configuration ---
print("Setting up LoRA configuration...")
peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    bias="none", # Usually set to 'none' for LoRA
    task_type="CAUSAL_LM",
)
# Apply LoRA adapters to the model
# model = get_peft_model(model, peft_config) # Apply PEFT config *before* trainer in recent versions? Check docs.

# --- Load Dataset ---
print(f"Loading dataset {dataset_id} from Hugging Face Hub...")
# Ensure you have network access and are logged in if it's private
dataset = load_dataset(dataset_id, split="train")
print(f"Dataset loaded: {len(dataset)} examples")

# --- Configure Training Arguments ---
print("Configuring training arguments using SFTConfig...")
# Use SFTConfig instead of TrainingArguments
training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=False, # QLoRA uses bfloat16 compute specified in bnb_config
    bf16=True, # Enable bfloat16 mixed precision (if GPU supports it)
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True, # Group sequences of similar length for efficiency
    lr_scheduler_type=lr_scheduler_type,
    gradient_checkpointing=gradient_checkpointing,
    report_to="tensorboard", # Or "wandb" if you have it configured
    push_to_hub=True, # Attempt to push the final model to the Hub
    hub_model_id=new_model_id, # Repository ID for the Hub
    hub_strategy="checkpoint", # Push checkpoints ('end', 'every_save', 'checkpoint')
    # hub_token=os.environ.get("HF_TOKEN") # Optionally pass token if needed
    max_seq_length=512, # Should be valid arg for SFTConfig
    dataset_text_field="text", # Explicitly set, even if it should be default
    packing=False,             # Can also be set here if needed (default False)
)

# --- Initialize Trainer ---
print("Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    args=training_args, # Pass the SFTConfig object here
    train_dataset=dataset,
    peft_config=peft_config, # Pass LoRA config here
    # tokenizer=tokenizer, # Removed: Trainer likely infers this from the model
    # dataset_text_field="text", # Already handled by SFTConfig default
    # max_seq_length=512,        # Now handled by SFTConfig
    # packing=False,             # Now handled by SFTConfig
)

# --- Start Training ---
print("Starting training...")
trainer.train()
print("Training finished.")

# --- Save Final Model Locally and Push (if configured) ---
print(f"Saving final model & tokenizer locally to {output_dir}...")
trainer.save_model(output_dir) # Saves LoRA adapters and config
# Ensure tokenizer is also saved
tokenizer.save_pretrained(output_dir)
print("Model saving complete.")

# The push_to_hub=True in TrainingArguments should handle the upload
# If you need manual push:
# print(f"Pushing final model to Hugging Face Hub: {new_model_id}")
# trainer.model.push_to_hub(new_model_id, config=peft_config, token=os.environ.get("HF_TOKEN"))
# tokenizer.push_to_hub(new_model_id, token=os.environ.get("HF_TOKEN"))
# print("Hub push complete.")

print("Script finished successfully!") 