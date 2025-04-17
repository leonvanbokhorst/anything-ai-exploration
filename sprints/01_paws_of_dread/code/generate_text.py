import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import warnings
import os

# Suppress specific warnings if needed (optional)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.generation.configuration_utils")

# --- Configuration ---
base_model_id = "microsoft/Phi-3-mini-4k-instruct"
adapter_model_id = "leonvanbokhorst/phi-3-mini-4k-instruct-existential-cats-v1" # Our fine-tuned adapters

# --- Load Tokenizer ---
print(f"Loading tokenizer for {base_model_id}...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # Ensure pad token is set
print(f"Using EOS token: '{tokenizer.eos_token}' with ID: {tokenizer.eos_token_id}")
print(f"Using PAD token: '{tokenizer.pad_token}' with ID: {tokenizer.pad_token_id}")

# --- Configure Quantization (optional, but recommended for consistency/efficiency) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# --- Load Base Model with Quantization ---
print(f"Loading base model {base_model_id} with 4-bit quantization...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto", # Automatically map layers to available devices
    trust_remote_code=True,
    # attn_implementation="flash_attention_2" # Optional for faster inference if installed
)
print("Base model loaded.")

# --- Load LoRA Adapters --- (No need to load separately with PeftModel.from_pretrained)
# print(f"Loading LoRA adapters from {adapter_model_id}...")
# No, PeftModel handles loading the adapters correctly on top.

# --- Apply Adapters to Base Model ---
print(f"Applying LoRA adapters from {adapter_model_id}...")
# This automatically loads the adapters from the Hub and applies them
model = PeftModel.from_pretrained(base_model, adapter_model_id)
print("Adapters applied. Model ready for inference.")

# --- Set Model to Evaluation Mode ---
model.eval()

# --- Generation Function ---
def generate_response(prompt, max_new_tokens=250, temperature=0.7, top_p=0.9):
    """Generates a response from the model given a prompt."""
    print(f"\n--- Prompt: ---\n{prompt}\n-----------------")

    # Prepare input for the model
    # Phi-3 instruct format is specific: <|user|>\nPROMPT<|end|>\n<|assistant|>\
    # We'll apply this basic structure for generation
    input_text = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
    inputs = tokenizer(input_text, return_tensors="pt", return_attention_mask=True).to(model.device)

    # Generate text
    with torch.no_grad(): # No need to track gradients for inference
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True, # Use sampling for more creative outputs
            pad_token_id=tokenizer.pad_token_id, # Important for generation
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode the generated tokens
    # We only want the response part, after the prompt and assistant token
    generated_sequence = outputs[0]
    input_length = inputs["input_ids"].shape[1]
    response_tokens = generated_sequence[input_length:]
    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

    print(f"\n--- Response: ---\n{response_text}\n-----------------")
    return response_text

# --- Scripted Generation ---
def run_scripted_generation(prompts_to_run, output_file=None):
    """Runs generation for a list of prompts and optionally saves to a file."""
    print("\n--- Running Scripted Generation ---")
    results = []

    # --- > Delete existing results file if it exists < --- #
    if output_file and os.path.exists(output_file):
        print(f"Deleting existing results file: {output_file}")
        try:
            os.remove(output_file)
        except Exception as e:
            print(f"Warning: Could not delete existing results file: {e}")
    # --- End of delete logic --- #

    for prompt in prompts_to_run:
        response = generate_response(prompt)
        results.append({"prompt": prompt, "response": response})

    if output_file:
        print(f"\nSaving results to {output_file}...")
        try:
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                print(f"Creating output directory: {output_dir}")
                os.makedirs(output_dir)

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# Existential Cat Meme Generations\n\n")
                for item in results:
                    f.write(f"## Prompt:\n```\n{item['prompt']}\n```\n\n")
                    f.write(f"### Response:\n```\n{item['response']}\n```\n\n---\n\n")
            print("Results saved.")
        except Exception as e:
            print(f"Error saving results to {output_file}: {e}")

    print("\n--- Scripted Generation Complete ---")


if __name__ == "__main__":
    # Define the prompts
    script_prompts = [
        "What is the meaning of the nap?",
        "Explain the empty food bowl.",
        "Why do I chase the laser I can never catch?",
        "Is the box my friend, or my prison?",
        "To meow, or not to meow?",
        "Tell me about the greebles.",
        "If I fits, I sits. But what if... the void fits?",
        "Describe the human's purpose.",
    ]

    # Define the output file path
    results_file = "sprints/01_paws_of_dread/results/generated_responses.md"

    # Run the generation
    run_scripted_generation(script_prompts, results_file)

    print("\nExiting generation script.")

# --- Interactive Loop (Commented out) ---
# if __name__ == "__main__":
#     print("\nStarting interactive generation. Type 'quit' or 'exit' to stop.")
#     while True:
#         try:
#             user_prompt = input("Enter your prompt: ")
#             if user_prompt.lower() in ["quit", "exit"]:
#                 break
#             if not user_prompt:
#                 continue
#             generate_response(user_prompt)
#         except EOFError:
#             break # Exit gracefully if input stream closes
#         except KeyboardInterrupt:
#             break # Exit gracefully on Ctrl+C
# 
#     print("\nExiting generation script.") 