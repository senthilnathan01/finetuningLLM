"""
Lab Utilities: Inspecting Finetuned vs. Base Models
Essential utility functions needed for the lab infrastructure.
Students will implement the evaluation functions as exercises.
"""

import os
import torch
import gc
import pandas as pd
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi

# =============================================================================
# HUGGING FACE UTILITIES
# =============================================================================

def validate_token():
    """
    Validate the current Hugging Face token and display user information.
    
    This function should be called after setting up your token to ensure
    it was configured correctly.
    """
    try:
        # Try to get token from environment variable first
        token = None
        for env_var in ['HUGGINGFACE_HUB_TOKEN', 'HF_TOKEN']:
            token = os.environ.get(env_var)
            if token:
                break
        
        if token:
            # Set the token for the API
            api = HfApi(token=token)
        else:
            # Try without explicit token (maybe already logged in)
            api = HfApi()
            
        user_info = api.whoami()
        print(f"✅ Token validated successfully!")
        print(f"   Logged in as: {user_info['name']}")
        print(f"   Token type: {user_info.get('type', 'unknown')}")
        return True
    except Exception as e:
        print(f"❌ Token validation failed.")
        print(f"   Error: {e}")
        print("\n💡 Please check that:")
        print("   1. Your token is correctly set")
        print("   2. Your token has the necessary permissions") 
        print("   3. You have access to the Deepseek models")
        print("   4. Try running: huggingface-cli login")
        return False

# =============================================================================
# MODEL SERVICE CLASS
# =============================================================================

class ServeLLM:
    """
    A service class for loading and running language models with proper memory management.
    """
    
    def __init__(self, model_name: str, device: str = "auto"):
        """
        Initialize the ServeLLM instance.
        
        Args:
            model_name (str): Name/path of the model to load
            device (str): Device to load model on ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.device = self._determine_device(device)
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _determine_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            # Check for CUDA (NVIDIA) or ROCm (AMD) availability
            if torch.cuda.is_available():
                return "cuda"
            # ROCm also uses torch.cuda API
            return "cpu"
        return device
    
    def _load_model(self):
        """Load the tokenizer and model."""
        try:
            print(f"Loading {self.model_name}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate settings
            model_kwargs = {
                "trust_remote_code": True,
                "dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "local_files_only": True
            }
            
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            print(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def generate_response(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True, 
    ) -> str:
        """
        Generate a response to the given prompt.
        
        Args:
            prompt (str): Input prompt
            max_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Top-p sampling parameter
            do_sample (bool): Whether to use sampling
            
        Returns:
            str: Generated response
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response (exclude input tokens)
            response_tokens = outputs[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def cleanup(self):
        """Clean up model and free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        print("🧹 Model cleaned up and memory freed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.cleanup()
    
    @staticmethod
    def cleanup_all():
        """Static method to clean up GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("🧹 All GPU memory cleaned up")

# =============================================================================
# DISPLAY UTILITIES
# =============================================================================

def display_section_header(title: str, level: int = 1):
    """
    Display a section header with appropriate formatting.
    
    Args:
        title (str): Section title
        level (int): Header level (1, 2, or 3)
    """
    if level == 1:
        print(f"\n{'='*60}")
        print(f"{title.upper()}")
        print('='*60)
    elif level == 2:
        print(f"\n{'-'*40}")
        print(f"{title}")
        print('-'*40)
    else:
        print(f"\n{title}")
        print('·'*len(title))

def display_warning(message: str):
    """Display a warning message in a prominent way."""
    print("⚠️  WARNING:", message)

def display_success(message: str):
    """Display a success message."""
    print("✅", message)

def display_info(message: str):
    """Display an info message."""
    print("ℹ️", message)