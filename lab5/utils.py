# utils.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import time
import os
import pandas as pd
from IPython.display import display

def setup_model_and_tokenizer(model_name="/app/models/llama-3.2-8b"):
    """Load and configure the model and tokenizer"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print(f"\nLoading {model_name}...")
    start_time = time.time()
    
    is_local = os.path.exists(model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        local_files_only=is_local  
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=True,
        local_files_only=is_local 
    )
    
    model.eval()
    load_time = time.time() - start_time
    print(f"Model loaded successfully in {load_time:.1f} seconds!")
    return model, tokenizer

def load_gsm8k_dataset(file_path='/app/data/GSM8K_dataset.jsonl'):
    """Load the GSM8K dataset"""
    problems = []
    with open(file_path, 'r') as f:
        for line in f:
            problems.append(json.loads(line.strip()))
    return problems

def save_results(results, filename):
    """Save results to JSON file"""
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")

def display_evaluation_results(json_file: str, num_rows: int = 15):
    """Display evaluation results from JSON file as a formatted table"""

    with open(json_file, "r") as f:
        data = json.load(f)
    
    summary_data = []
    for i, result in enumerate(data):
        problem_num = i + 1
        ground_truth = result['ground_truth']
        
        for solution in result['solutions']:
            row = {
                'Problem': problem_num,
                'Ground Truth': ground_truth,
                'Template': solution['template'].upper(),
                'Rank': solution['rank'],
                'Composite Score': solution['composite_score'],
                **solution['scores']
            }
            summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    print("Evaluation Results Summary")
    print("=" * 80)
    display(df.head(num_rows))
    
    return df