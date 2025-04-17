import os
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login, HfApi


def load_env_vars(env_path: Optional[str] = None) -> Dict[str, str]:
    """
    Load environment variables from .env file
    
    Args:
        env_path: Path to .env file. If None, will look in current directory
                 and parent directories.
    
    Returns:
        Dictionary of environment variables
    """
    if env_path is None:
        # Find .env file in current directory or parent directories
        current_dir = Path.cwd()
        env_path = current_dir / ".env"
        
        # If not in current dir, try parent directories (up to 3 levels)
        if not env_path.exists():
            for _ in range(3):
                current_dir = current_dir.parent
                env_path = current_dir / ".env"
                if env_path.exists():
                    break
    
    # Load environment variables
    load_dotenv(env_path)
    
    # Return dict of relevant API keys
    return {
        "hf_token": os.getenv("HUGGINGFACE_API_KEY"),
        "hf_username": os.getenv("HF_USERNAME"),
        "wandb_token": os.getenv("WANDB_API_KEY")
    }


def login_huggingface(token: Optional[str] = None) -> None:
    """
    Login to Hugging Face with API token
    
    Args:
        token: Hugging Face API token. If None, will use HUGGINGFACE_API_KEY 
               from environment variables.
    """
    if token is None:
        token = os.getenv("HUGGINGFACE_API_KEY")
        
    if not token:
        raise ValueError(
            "Hugging Face API token not found. Please set HUGGINGFACE_API_KEY "
            "environment variable or provide token directly."
        )
    
    # Login to Hugging Face
    login(token=token, add_to_git_credential=True)
    print("Successfully logged in to Hugging Face!")


def get_huggingface_api() -> HfApi:
    """
    Get authenticated Hugging Face API client
    
    Returns:
        Authenticated HfApi instance
    """
    # Ensure we're logged in
    token = os.getenv("HUGGINGFACE_API_KEY")
    if not token:
        raise ValueError("Hugging Face API token not found. Please call load_env_vars() first.")
    
    return HfApi(token=token)


if __name__ == "__main__":
    # Example usage
    env_vars = load_env_vars()
    print(f"Found HF username: {env_vars['hf_username']}")
    
    # Login to Hugging Face
    login_huggingface(env_vars["hf_token"]) 