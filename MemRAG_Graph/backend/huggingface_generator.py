from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from typing import List, Dict 

logger = logging.getLogger('RAG')


class HuggingfaceGenerator:
    """
    Generator module using models like Qwen2.5 Instruct models from Huggingface.
    Handles prompt formatting and answer generation.
    Can be extended for Feature B (Agentic Workflow).
    """
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        """
        Initializes the Qwen Generator.

        Args:
            model_name (str): The name of the Qwen model to use.
        """
        self.model_name = model_name
        logger.info(f"Loading Qwen model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        use_cuda = torch.cuda.is_available()
        dtype = torch.float16 if use_cuda else torch.float32
        device = "cuda" if use_cuda else "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=None,          
            low_cpu_mem_usage=not use_cuda
        ).to(device)
        self.model.eval()
        logger.info("Qwen model loaded successfully.")


    def generate(self, prompt: str) -> str:
        """
        Generates an answer based on the query only.

        Args:
            prompt (str): The final prompt
        Returns:
            str: The answer
        """
        messages = [
            { "role" : "system", "content" : "You are a helpful assistant. Answer strictly using the provided context and include citations when requested."},
            { "role": "user", "content": prompt }
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=256, # cap output to reduce drift
            do_sample=False, # Deterministic for consistency
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Decode only the newly generated part
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        logger.info("Generation completed.")
        return response
