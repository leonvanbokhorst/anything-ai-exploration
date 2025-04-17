#!/usr/bin/env python3
"""
Test script for authentication utilities
"""
from utils.auth import load_env_vars, login_huggingface

def main():
    """Test authentication utilities"""
    print("Loading environment variables...")
    env_vars = load_env_vars()
    
    # Print found credentials (without revealing full keys)
    if env_vars["hf_token"]:
        masked_token = env_vars["hf_token"][:4] + "..." + env_vars["hf_token"][-4:]
        print(f"Found HF token: {masked_token}")
    else:
        print("No HF token found")
        
    print(f"Found HF username: {env_vars['hf_username']}")
    
    if env_vars["wandb_token"]:
        masked_wandb = env_vars["wandb_token"][:4] + "..." + env_vars["wandb_token"][-4:]
        print(f"Found Wandb token: {masked_wandb}")
    
    # Try logging in to Hugging Face
    print("\nTrying Hugging Face login...")
    try:
        login_huggingface(env_vars["hf_token"])
        print("🎉 Authentication test successful! You're ready to work with Hugging Face models.")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        
if __name__ == "__main__":
    main() 