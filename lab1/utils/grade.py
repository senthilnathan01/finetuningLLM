from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from serve_llm import ServeLLM
from grade_reference import *

base_model = "deepseek-ai/deepseek-math-7b-base"
sft_model = "deepseek-ai/deepseek-math-7b-instruct"
rl_model = "deepseek-ai/deepseek-math-7b-rl"


def calculate_similarity_scores(base_model_ans: str, sft_model_ans: str, rl_model_ans: str):
    """
    Calculate cosine similarity scores between model answers using sentence embeddings.
    
    Args:
        base_model_ans: Answer from base model
        sft_model_ans: Answer from SFT model  
        rl_model_ans: Answer from RL model
        
    Returns:
        tuple: (base_rl_similarity, sft_rl_similarity)
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Calculate the Cosine Similarity of Sentence Embeddings for base model and sft model
    embeddings = model.encode([base_model_ans, sft_model_ans, rl_model_ans])
    base_rl_similarity = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]
    sft_rl_similarity = cosine_similarity([embeddings[1]], [embeddings[2]])[0][0]

    return base_rl_similarity, sft_rl_similarity


def section_4_grading_correctness_results(num_samples=20, min_expected_acc=0.2):
    """
    return pass/fail status
    """
    grade = {
        "extract_number": test_extract_number(),
        "model_results": {},
        "overall_pass": True
    }

    results = evaluate_all_models(num_samples=num_samples)
    grade["model_results"] = results

    for model_name, acc in results.items():
        if acc is None or acc < min_expected_acc:
            grade["overall_pass"] = False

    return grade


def section_3_grading_prompt_modification(base_model_prompt: str, sft_model_prompt: str):
    
    llm = ServeLLM(base_model)
    base_model_ans = llm.generate_response(base_model_prompt, temperature=0.0, top_p=1.0, max_tokens=5000)
    print(f"\n\n\nBase model answer: {base_model_ans} \n")
    llm.cleanup()

    llm = ServeLLM(sft_model)
    sft_model_ans = llm.generate_response(sft_model_prompt, temperature=0.0, top_p=1.0, max_tokens=5000)
    print(f"\nSFT model answer: {sft_model_ans} \n\n\n")

    llm.cleanup()

    # Use the extracted function to calculate similarity scores
    base_rl_similarity, sft_rl_similarity = calculate_similarity_scores(base_model_ans, sft_model_ans, rl_model_ans)

    return base_rl_similarity, sft_rl_similarity


def section_3_grading_reference():
    print("Base model and SFT model are using the same prompt of the RL model")

    base_model_prompt = original_rl_model_prompt
    sft_model_prompt = original_rl_model_prompt
    base_rl_similarity, sft_rl_similarity = section_3_grading_prompt_modification(base_model_prompt, sft_model_prompt)

    # Base RL Similarity: 0.17203107476234436
    # SFT RL Similarity: 0.6431516408920288


    print(f"Base RL Similarity: {base_rl_similarity}")
    print(f"SFT RL Similarity: {sft_rl_similarity}")

    # ------------------------------------------------------------
    
    print("Base model and SFT model are using the reference prompt of the RL model")
    
    base_model_prompt = base_model_reference_prompt
    sft_model_prompt = sft_model_reference_prompt
    base_rl_similarity, sft_rl_similarity = section_3_grading_prompt_modification(base_model_prompt, sft_model_prompt)


    # Base RL Similarity: 0.6091499328613281
    # SFT RL Similarity: 0.6665745377540588

    print(f"Base RL Similarity: {base_rl_similarity}")
    print(f"SFT RL Similarity: {sft_rl_similarity}")


if __name__ == "__main__":
    # similarity_test()
    # section_3_grading_reference()
    pass

