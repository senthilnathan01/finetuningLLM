"""
Utility functions for the LLM Evaluation and Debugging Lab
"""

import time
import re
from scipy import stats
import shutil
import os
from pathlib import Path
import matplotlib.pyplot as plt
import json
import pickle
import hashlib
from typing import Dict, List, Any, Optional
import torch 
import warnings
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, GenerationConfig, DataCollatorForLanguageModeling
from dotenv import load_dotenv

# Try to load .env file from current directory, then from parent directories
env_loaded = False
current_dir = Path.cwd()
for _ in range(3):  # Check current dir and up to 2 parent directories
    env_path = current_dir / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        env_loaded = True
        break
    current_dir = current_dir.parent

# If no .env file found, try loading from default location
if not env_loaded:
    load_dotenv()

# HF_TOKEN is optional since models are cached in the Docker image
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# Set up environment
warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# Global variables for timing functions
section_timings = {}
current_section_start = None
# Remove specified folders
folders_to_remove = [
    os.path.join(Path(os.path.dirname(__file__)).parent, "training_logs"),
    os.path.join(Path(os.path.dirname(__file__)).parent, "lab_results"),
]

for folder in folders_to_remove:
    folder_path = os.path.join(os.getcwd(), folder)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
       
def start_timer(section_name: str):
    """Start timing a section."""
    global current_section_start
    current_section_start = time.time()
    print(f"⏱️  Started: {section_name}")

def end_timer(section_name: str):
    """End timing a section."""
    global current_section_start, section_timings
    if current_section_start is not None:
        duration = time.time() - current_section_start
        section_timings[section_name] = duration
        print(f"✅ Completed in {duration:.1f}s: {section_name}")
        current_section_start = None

def get_section_timings():
    """Get the dictionary of section timings."""
    return section_timings


# Add extract function from solutions.ipynb for proper answer extraction
def extract_numerical_answer(ans: str) -> str:
    """Extract the final numerical answer from plaintext (from solutions.ipynb)."""
    ind = ans.rindex('\n') + 1
    ans_marker = '#### '
    if ans[ind: ind + len(ans_marker)] != ans_marker:
        raise ValueError(
            f'Incorrectly formatted answer `{ans}` does not have '
            f'`{ans_marker}` at the end'
        )
    return ans[ind + len(ans_marker):]

def extract_final_answer(response: str) -> str:
    """Extract the final numerical answer from model response."""
    # Look for patterns like "The answer is X" or numbers at the end
    patterns = [
        r"The answer is ([+-]?\d+\.?\d*)",
        r"= ([+-]?\d+\.?\d*)",
        r"([+-]?\d+\.?\d*)\s*$"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response)
        if matches:
            return matches[-1].strip()
    
    # Fallback: extract last number
    numbers = re.findall(r'([+-]?\d+\.?\d*)', response)
    return numbers[-1] if numbers else "0"

def mcnemar_test(original_correct, improved_correct):
    """Simple McNemar's test for comparing two models."""
    # Count the different outcomes
    both_correct = sum(1 for o, i in zip(original_correct, improved_correct) if o and i)
    only_original = sum(1 for o, i in zip(original_correct, improved_correct) if o and not i)
    only_improved = sum(1 for o, i in zip(original_correct, improved_correct) if not o and i)
    both_wrong = sum(1 for o, i in zip(original_correct, improved_correct) if not o and not i)
    
    # McNemar's test
    if only_original + only_improved == 0:
        return 1.0  # No difference
    
    statistic = (abs(only_original - only_improved) - 1) ** 2 / (only_original + only_improved)
    p_value = 1 - stats.chi2.cdf(statistic, 1)
    return p_value

# Caching functionality
def create_cache_key(model_name: str, dataset_subset: List[str]) -> str:
    """Create a unique cache key based on model name and dataset questions."""
    # Use first 3 and last 3 questions to create a representative hash
    sample_questions = dataset_subset[:3] + dataset_subset[-3:] if len(dataset_subset) > 6 else dataset_subset
    combined_str = f"{model_name}:{':'.join(sample_questions)}"
    return hashlib.md5(combined_str.encode()).hexdigest()[:16]

def save_evaluation_cache(cache_key: str, results: Dict, cache_dir: str = "/app/data/cache/") -> None:
    """Save evaluation results to cache."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"eval_{cache_key}.pkl")
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"💾 Saved evaluation results to cache: {cache_file}")
    except Exception as e:
        print(f"⚠️ Failed to save evaluation cache: {e}")

def load_evaluation_cache(cache_key: str, cache_dir: str = "/app/data/cache/") -> Optional[Dict]:
    """Load evaluation results from cache."""
    cache_file = os.path.join(cache_dir, f"eval_{cache_key}.pkl")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                results = pickle.load(f)
            print(f"📂 Loaded evaluation results from cache: {cache_file}")
            return results
        except Exception as e:
            print(f"⚠️ Failed to load evaluation cache: {e}")
            return None
    return None

def save_error_analysis_cache(cache_key: str, error_analyses: List[Dict], cache_dir: str = "/app/data/cache/") -> None:
    """Save error analysis results to cache."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"error_{cache_key}.json")
    
    try:
        with open(cache_file, 'w') as f:
            json.dump(error_analyses, f, indent=2)
        print(f"💾 Saved error analysis results to cache: {cache_file}")
    except Exception as e:
        print(f"⚠️ Failed to save error analysis cache: {e}")

def load_error_analysis_cache(cache_key: str, cache_dir: str = "/app/data/cache/") -> Optional[List[Dict]]:
    """Load error analysis results from cache."""
    cache_file = os.path.join(cache_dir, f"error_{cache_key}.json")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                error_analyses = json.load(f)
            print(f"📂 Loaded error analysis results from cache: {cache_file}")
            return error_analyses
        except Exception as e:
            print(f"⚠️ Failed to load error analysis cache: {e}")
            return None
    return None

def cached_evaluate_model(evaluate_model_func, model, tokenizer, dataset, model_name: str) -> Dict:
    """
    Cached version of evaluate_model function.
    
    Args:
        evaluate_model_func: The actual evaluate_model function to call if cache miss
        model: The model to evaluate
        tokenizer: The tokenizer to use
        dataset: The dataset to evaluate on
        model_name: Name of the model for cache identification
        
    Returns:
        Dict: Evaluation results (from cache or fresh computation)
    """
    # Create cache key from questions in the dataset
    questions = [example['question'] for example in dataset]
    cache_key = create_cache_key(model_name, questions)
    
    # Try to load from cache first
    cached_results = load_evaluation_cache(cache_key)
    if cached_results is not None:
        print(f"🚀 Using cached evaluation results for {model_name}")
        return cached_results
    
    # If not in cache, run evaluation and save results
    print(f"🔄 Cache miss - running fresh evaluation for {model_name}")
    results = evaluate_model_func(model, tokenizer, dataset, model_name)
    
    # Save to cache
    save_evaluation_cache(cache_key, results)
    
    return results

def cached_analyze_errors(analyze_error_func, questions: List[str], correct_answers: List[str], 
                         model_responses: List[str], predicted_answers: List[str]) -> List[Dict]:
    """
    Cached version of error analysis function.
    
    Args:
        analyze_error_func: The actual analyze_error function to call if cache miss
        questions: List of questions that had errors
        correct_answers: List of correct answers for error questions
        model_responses: List of model responses for error questions
        predicted_answers: List of predicted answers for error questions
        
    Returns:
        List[Dict]: Error analysis results (from cache or fresh computation)
    """
    # Create cache key from error data
    error_data = f"errors:{len(questions)}:{':'.join(questions[:3])}"  # Sample for key
    cache_key = hashlib.md5(error_data.encode()).hexdigest()[:16]
    
    # Try to load from cache first
    cached_results = load_error_analysis_cache(cache_key)
    if cached_results is not None:
        print(f"🚀 Using cached error analysis results ({len(cached_results)} errors)")
        return cached_results
    
    error_analyses = []
    
    for i, (question, correct_answer, model_response, predicted_answer) in enumerate(
        zip(questions, correct_answers, model_responses, predicted_answers)
    ):
        analysis = analyze_error_func(question, correct_answer, model_response, predicted_answer)
        analysis['example_index'] = i  # Keep track of original index
        error_analyses.append(analysis)
        
        if (i + 1) % 10 == 0:
            print(f"Analyzed {i + 1}/{len(questions)} errors")
    
    # Save to cache
    save_error_analysis_cache(cache_key, error_analyses)
    
    return error_analyses

def plot_training_losses(log_dir: str, figsize=(12, 6)) -> None:
    """
    Plot training and validation losses from TensorBoard logs.
    
    Args:
        log_dir (str): Path to TensorBoard log directory
        figsize (tuple): Figure size for the plot
    """
    import os


def get_model(model_name):
    # Use cached models if available (set by Docker environment variables)
    # Priority: 1) HF_HOME env var, 2) /app/models (Docker), 3) local models/ dir
    cache_dir = os.getenv('HF_HOME', None)
    
    # If no cache_dir from env, check common locations for pre-downloaded models
    if cache_dir is None:
        potential_dirs = [
            '/app/models',  # Docker location
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')  # Local models/ dir
        ]
        for potential_dir in potential_dirs:
            if os.path.exists(potential_dir) and os.path.isdir(potential_dir):
                cache_dir = potential_dir
                print(f"📦 Using local model cache: {cache_dir}")
                break
    
    # Load from local cache only - do not download from internet
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map='auto', cache_dir=cache_dir, local_files_only=True)

    # Specify beginning of sequence token, end of sequence token, and padding token
    model.generation_config = GenerationConfig.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    return model, tokenizer


def generate_batch_responses(model, tokenizer, questions: List[str], batch_size: int = 8) -> List[str]:
    """Generate model responses for multiple questions using batch processing."""
    responses = []
    
    # Process questions in batches
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i + batch_size]
        
        # Format prompts
        batch_prompts = [f"Problem: {question}\nSolution:" for question in batch_questions]
        
        # Tokenize
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate responses
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode responses
        for j, output in enumerate(outputs):
            input_length = inputs['input_ids'][j].shape[0]
            generated_tokens = output[input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response.strip())
        
        print(f"Processed batch {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}")
    
    return responses


# Error analysis constants and functionality
ERROR_CATEGORIES = [
    "calculation_error",   # Math computation mistakes  
    "reasoning_error",     # Logical reasoning issues
    "incomplete_solution", # Partial or unfinished answers
    "format_error",       # Wrong answer format
    "other"              # Other errors
]

def load_error_analysis_model():
    """
    Load the error analysis model from local disk.
    Returns:
        tuple: (tokenizer, model, hf_model_available) where hf_model_available is bool
    """
    print("Loading local model for error analysis...")
    try:
        # Define the local model path (matching the actual directory name)
        local_model_path = '/app/models/llama-3.2-8b'
        
        # Verify the model directory exists
        if not os.path.exists(local_model_path):
            raise FileNotFoundError(f"Model directory not found at: {local_model_path}")
        
        if not os.path.isdir(local_model_path):
            raise NotADirectoryError(f"Path exists but is not a directory: {local_model_path}")
        
        print(f"📦 Loading model from: {local_model_path}")
        
        # Load tokenizer from local path only
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            local_model_path,
            local_files_only=True,
            trust_remote_code=True
        )
        
        # Load model from local path only
        print("Loading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            local_files_only=True
        )
        
        # Set pad token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Local model loaded successfully!")
        return tokenizer, model, True
        
    except Exception as e:
        print(f"⚠️ Could not load local model: {e}")
        print("Will use rule-based fallback analysis")
        return None, None, False


def cluster_errors(error_analyses: List[Dict], questions: List[str]) -> tuple:
    """
    Cluster similar errors using sentence transformers and K-means clustering.
    
    Args:
        error_analyses: List of error analysis dictionaries
        questions: List of questions corresponding to the errors
        
    Returns:
        tuple: (embeddings, cluster_labels, embedding_model, kmeans)
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans

    print("Loading sentence transformer for clustering...")
    # Use local model path instead of downloading from internet
    local_model_path = '/app/models/all-MiniLM-L6-v2'
    if os.path.exists(local_model_path):
        print(f"Loading sentence transformer from local path: {local_model_path}")
        embedding_model = SentenceTransformer(local_model_path)
    else:
        print(f"Local model not found at {local_model_path}, falling back to download")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create embeddings for the error text
    print("Creating embeddings...")
    texts = []
    for analysis in error_analyses:
        idx = analysis['example_index']
        # Combine question and error info
        text = f"{questions[idx]} {analysis['category']} {analysis['description']}"
        texts.append(text)

    embeddings = embedding_model.encode(texts)

    # Cluster the errors
    n_clusters = min(3, len(error_analyses) // 10 + 1)  # Keep it simple
    print(f"Clustering into {n_clusters} groups...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    
    return embeddings, cluster_labels, embedding_model, kmeans, n_clusters