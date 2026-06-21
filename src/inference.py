"""Inference engine for generating model responses."""
import time
from typing import List, Dict, Any
from dataclasses import dataclass

import torch
from tqdm import tqdm
from loguru import logger

from .config import Config


@dataclass
class GenerationResult:
    prompt: str
    response: str
    language: str
    prompt_id: str
    generation_time: float = 0.0


class InferenceEngine:
    """Handles text generation."""
    
    def __init__(self, model, tokenizer, config: Config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = model.device
        
        # Enable inference optimizations
        self.model.eval()
        if hasattr(torch, 'inference_mode'):
            self._inference_context = torch.inference_mode
        else:
            self._inference_context = torch.no_grad
    
    def _format_prompt(self, prompt: str) -> str:
        """Format for Gemma-2 chat template."""
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    
    @torch.inference_mode()
    def generate(self, prompt: str, language: str = "en", prompt_id: str = "0") -> GenerationResult:
        """Generate response for a single prompt."""
        formatted = self._format_prompt(prompt)
        
        inputs = self.tokenizer(
            formatted, return_tensors="pt", padding=True,
            truncation=True, max_length=1024
        ).to(self.device)
        
        start = time.time()
        
        # Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "do_sample": self.config.do_sample,
        }
        
        # Only add sampling params if do_sample is True
        if self.config.do_sample:
            gen_kwargs["temperature"] = self.config.temperature
            gen_kwargs["top_p"] = self.config.top_p
        
        outputs = self.model.generate(**inputs, **gen_kwargs)
        gen_time = time.time() - start
        
        input_len = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        response = response.split("<end_of_turn>")[0].strip()
        
        return GenerationResult(prompt, response, language, prompt_id, gen_time)
    
    def generate_parallel(self, prompt_pairs: List[Dict], show_progress: bool = True) -> List[Dict[str, Any]]:
        """Generate responses for English-Bengali pairs."""
        results = []
        total_time = 0
        
        iterator = tqdm(prompt_pairs, desc="Generating") if show_progress else prompt_pairs
        
        for idx, pair in enumerate(iterator):
            en = self.generate(pair["en"], "en", f"{idx}_en")
            bn = self.generate(pair["bn"], "bn", f"{idx}_bn")
            
            total_time += en.generation_time + bn.generation_time
            avg_time = total_time / ((idx + 1) * 2)
            
            results.append({
                "id": idx,
                "en_prompt": pair["en"],
                "bn_prompt": pair["bn"],
                "en_response": en.response,
                "bn_response": bn.response,
                "category": pair.get("category", "unknown"),
                "en_time": en.generation_time,
                "bn_time": bn.generation_time
            })
            
            if show_progress and (idx + 1) % 10 == 0:
                remaining = (len(prompt_pairs) - idx - 1) * 2 * avg_time
                logger.info(f"Avg: {avg_time:.1f}s/response | ETA: {remaining/60:.1f}min")
        
        logger.info(f"Generated {len(results)} pairs in {total_time/60:.1f} minutes")
        return results
