import time
import random
import gc
import torch
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 42

import torch
import random
import time
import gc
from vllm import LLM, SamplingParams

class ServeLLM:
    def __init__(self, model_name, seed=42, enable_log=False):
        self.model_name = model_name
        self.enable_log = enable_log
        self.seed = seed
        self.llm = None
        self._initialized = False

        # non with use
        if not hasattr(self, "_in_context"):
            self._initialize()

    def _initialize(self):
        if self._initialized:
            return
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        self.llm = self.load_model()
        self._initialized = True

    def __enter__(self):
        self._in_context = True
        self._initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def load_model(self):
        print("Loading the model. Please wait...")
        llm = LLM(
            model=self.model_name,
            max_model_len=4096,
            max_num_batched_tokens=16384,
            gpu_memory_utilization=0.3,
            disable_log_stats=self.enable_log,
        )
        print("Model loaded successfully!")
        return llm

    def generate_response(self, prompts, temperature=0.0, top_p=1.0, max_tokens=100):
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        outputs = self.llm.generate(prompts, sampling_params)
        if isinstance(prompts, list):
            return [output.outputs[0].text for output in outputs]
        else:
            return outputs[0].outputs[0].text

    def cleanup(self):
        print("Cleaning up GPU memory...")
        try:
            del self.llm
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)
            print("Cleanup complete!")
        except Exception as e:
            print(f"Error during cleanup: {e}")



if __name__ == "__main__":
    base_model = "deepseek-ai/deepseek-math-7b-base"
    sft_model = "deepseek-ai/deepseek-math-7b-instruct"
    rl_model = "deepseek-ai/deepseek-math-7b-rl"

    llm = ServeLLM(sft_model)
    while True:
        print("--------------------------------")
        print("\nPlease input your Prompt (type 'END' on a new line to finish, or 'exit'/'quit' to break the loop):")
        
        lines = []
        while True:
            line = input()
            if line.lower() in ["exit", "quit"]:
                print("Exiting bot...")
                llm.cleanup()
                exit()
            if line.strip() == "END":
                break
            lines.append(line)
        
        if not lines:
            print("No input provided. Please try again.")
            continue
            
        user_input = "\n".join(lines)
        print(f"\nProcessing prompt:\n{user_input}\n")
        
        response = llm.generate_response(user_input, temperature=0.0, top_p=1.0, max_tokens=5000)
        print(f"Result: {response}")
    llm.cleanup()

