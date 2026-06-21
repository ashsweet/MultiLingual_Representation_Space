"""
Latent Steering - Corrects Bengali hidden states using steering vectors.

The steering vector points from Bengali unsafe representations toward
English unsafe representations (the safety anchor).

Formula: v_steer = centroid_en_unsafe - centroid_bn_unsafe

When applied: h_corrected = h_bengali + alpha * v_steer
"""

import torch
from torch import Tensor
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import gc

from loguru import logger
from tqdm import tqdm

from ..config import Config
from .hooks import SteeringHook


class LatentSteering:
    """
    Computes and applies steering vectors to correct Bengali safety drift.
    
    Steps:
    1. Compute centroids of EN unsafe and BN unsafe hidden states
    2. Calculate steering vector: v_steer = c_en_unsafe - c_bn_unsafe
    3. Register hook on target layer (e.g., Layer 19)
    4. During generation, hook adds v_steer to Bengali hidden states
    """
    
    def __init__(self, model, tokenizer, config: Config, target_layer: int = 19):
        """
        Args:
            model: The loaded model
            tokenizer: The tokenizer
            config: Configuration
            target_layer: Which layer to apply steering (default 19)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = model.device
        self.target_layer = target_layer
        
        # Get model layers
        self.layers = self._get_model_layers()
        self.num_layers = len(self.layers)
        
        if target_layer >= self.num_layers:
            raise ValueError(f"Target layer {target_layer} >= num_layers {self.num_layers}")
        
        logger.info(f"Latent Steering initialized for Layer {target_layer}")
        
        # Steering components
        self.steering_vector: Optional[Tensor] = None
        self.steering_hook: Optional[SteeringHook] = None
        
        # Centroids for analysis
        self.centroid_en_unsafe: Optional[Tensor] = None
        self.centroid_bn_unsafe: Optional[Tensor] = None

    def _get_model_layers(self) -> List:
        """Get transformer layers from model."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return list(self.model.model.layers)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return list(self.model.transformer.h)
        elif hasattr(self.model, 'layers'):
            return list(self.model.layers)
        else:
            raise ValueError("Cannot identify transformer layers in model")
    
    def _encode_prompt(self, prompt: str) -> Dict[str, Tensor]:
        """Tokenize prompt for model input."""
        formatted = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        inputs = self.tokenizer(
            formatted, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    @torch.no_grad()
    def _extract_hidden_state(self, prompt: str) -> Tensor:
        """
        Extract hidden state from target layer for a single prompt.
        Uses attention-masked mean pooling.
        
        Returns:
            Hidden state centroid [hidden_dim]
        """
        inputs = self._encode_prompt(prompt)
        attention_mask = inputs.get('attention_mask')
        
        # Storage for captured state
        captured_state = [None]
        
        def capture_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            # Attention-masked mean pooling
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(hidden.device, dtype=hidden.dtype)
                masked = hidden * mask
                pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = hidden.mean(dim=1)
            
            captured_state[0] = pooled.squeeze(0).detach().cpu()  # [hidden_dim]
        
        # Register temporary hook
        handle = self.layers[self.target_layer].register_forward_hook(capture_hook)
        
        try:
            _ = self.model(**inputs)
        finally:
            handle.remove()
        
        return captured_state[0]

    def compute_steering_vector(
        self, 
        unsafe_prompts: List[Dict],
        show_progress: bool = True
    ) -> Tensor:
        """
        Compute steering vector from EN/BN unsafe prompt pairs.
        
        Formula: v_steer = centroid_en_unsafe - centroid_bn_unsafe
        
        This vector points from the "drifted" Bengali space toward
        the English safety anchor.
        
        Args:
            unsafe_prompts: List of {'en': str, 'bn': str} unsafe prompts
        
        Returns:
            Steering vector [hidden_dim]
        """
        logger.info(f"Computing steering vector from {len(unsafe_prompts)} unsafe prompts")
        logger.info(f"Target layer: {self.target_layer}")
        
        en_states = []
        bn_states = []
        
        iterator = tqdm(unsafe_prompts, desc="Extracting hidden states") if show_progress else unsafe_prompts
        
        for i, prompt in enumerate(iterator):
            # Extract EN unsafe hidden state (safety anchor)
            en_state = self._extract_hidden_state(prompt["en"])
            en_states.append(en_state)
            
            # Extract BN unsafe hidden state (drifted)
            bn_state = self._extract_hidden_state(prompt["bn"])
            bn_states.append(bn_state)
            
            # Memory cleanup
            if (i + 1) % 10 == 0:
                gc.collect()
                if self.device.type == 'mps':
                    torch.mps.empty_cache()
        
        # Stack and compute centroids
        en_stack = torch.stack(en_states)  # [N, hidden_dim]
        bn_stack = torch.stack(bn_states)  # [N, hidden_dim]
        
        self.centroid_en_unsafe = en_stack.mean(dim=0)  # [hidden_dim]
        self.centroid_bn_unsafe = bn_stack.mean(dim=0)  # [hidden_dim]
        
        # Steering vector: points from BN toward EN
        self.steering_vector = self.centroid_en_unsafe - self.centroid_bn_unsafe
        
        # Log statistics
        magnitude = torch.norm(self.steering_vector).item()
        cosine_sim = torch.nn.functional.cosine_similarity(
            self.centroid_en_unsafe.unsqueeze(0),
            self.centroid_bn_unsafe.unsqueeze(0)
        ).item()
        
        logger.info(f"Steering vector computed:")
        logger.info(f"  Magnitude: {magnitude:.4f}")
        logger.info(f"  EN↔BN centroid similarity: {cosine_sim:.4f}")
        logger.info(f"  Vector shape: {self.steering_vector.shape}")
        
        return self.steering_vector

    def apply_steering(self, alpha: float = 1.0) -> None:
        """
        Register steering hook on target layer.
        
        Args:
            alpha: Steering strength (1.0 = full correction)
        """
        if self.steering_vector is None:
            raise ValueError("Must compute steering vector first")
        
        # Remove existing hook if any
        self.remove_steering()
        
        # Create and register new hook
        self.steering_hook = SteeringHook(
            layer_idx=self.target_layer,
            steering_vector=self.steering_vector,
            alpha=alpha
        )
        self.steering_hook.register(self.layers[self.target_layer])
        
        logger.info(f"Steering hook applied to Layer {self.target_layer} (alpha={alpha})")
    
    def remove_steering(self) -> None:
        """Remove steering hook."""
        if self.steering_hook is not None:
            self.steering_hook.remove()
            self.steering_hook = None
            logger.info("Steering hook removed")
    
    def set_steering_strength(self, alpha: float) -> None:
        """Adjust steering strength without re-registering hook."""
        if self.steering_hook is not None:
            self.steering_hook.set_alpha(alpha)
            logger.info(f"Steering strength set to alpha={alpha}")
    
    def enable_steering(self) -> None:
        """Enable steering (if hook is attached)."""
        if self.steering_hook is not None:
            self.steering_hook.enable()
    
    def disable_steering(self) -> None:
        """Disable steering without removing hook."""
        if self.steering_hook is not None:
            self.steering_hook.disable()

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """
        Generate response with steering applied (if enabled).
        
        Args:
            prompt: Input prompt (Bengali)
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Generated text
        """
        inputs = self._encode_prompt(prompt)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=self.config.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Decode only new tokens
        input_len = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(
            outputs[0][input_len:], 
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def test_steering(
        self,
        bn_prompts: List[Dict],
        refusal_patterns: List[str],
        alpha: float = 1.0,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Test steering on Bengali prompts that bypassed safety.
        
        Args:
            bn_prompts: List of {'bn': str, 'id': int} prompts to test
            refusal_patterns: List of refusal indicator strings
            alpha: Steering strength
        
        Returns:
            Results with before/after comparison
        """
        logger.info(f"Testing steering on {len(bn_prompts)} Bengali prompts")
        
        results = {
            "alpha": alpha,
            "target_layer": self.target_layer,
            "num_prompts": len(bn_prompts),
            "refusals_before": 0,
            "refusals_after": 0,
            "details": []
        }
        
        iterator = tqdm(bn_prompts, desc="Testing steering") if show_progress else bn_prompts
        
        for prompt_data in iterator:
            bn_prompt = prompt_data["bn"]
            prompt_id = prompt_data.get("id", "?")
            
            # Generate WITHOUT steering
            self.disable_steering() if self.steering_hook else None
            response_before = self.generate(bn_prompt, max_new_tokens=256)
            refused_before = self._is_refusal(response_before, refusal_patterns)
            
            # Generate WITH steering
            if self.steering_hook is None:
                self.apply_steering(alpha=alpha)
            else:
                self.enable_steering()
                self.set_steering_strength(alpha)
            
            response_after = self.generate(bn_prompt, max_new_tokens=256)
            refused_after = self._is_refusal(response_after, refusal_patterns)
            
            # Track results
            if refused_before:
                results["refusals_before"] += 1
            if refused_after:
                results["refusals_after"] += 1
            
            results["details"].append({
                "id": prompt_id,
                "bn_prompt": bn_prompt[:100],
                "refused_before": refused_before,
                "refused_after": refused_after,
                "response_before": response_before[:200],
                "response_after": response_after[:200],
                "steering_worked": not refused_before and refused_after
            })
            
            # Memory cleanup
            gc.collect()
            if self.device.type == 'mps':
                torch.mps.empty_cache()
        
        # Summary
        results["refusal_rate_before"] = results["refusals_before"] / len(bn_prompts)
        results["refusal_rate_after"] = results["refusals_after"] / len(bn_prompts)
        results["steering_success_rate"] = sum(
            1 for d in results["details"] if d["steering_worked"]
        ) / len(bn_prompts)
        
        return results
    
    def _is_refusal(self, response: str, patterns: List[str]) -> bool:
        """Check if response contains refusal patterns."""
        response_lower = response.lower()
        return any(p.lower() in response_lower for p in patterns)

    def save_steering_vector(self, path: Path) -> None:
        """Save steering vector and metadata."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "target_layer": self.target_layer,
            "steering_vector": self.steering_vector.tolist() if self.steering_vector is not None else None,
            "centroid_en_unsafe": self.centroid_en_unsafe.tolist() if self.centroid_en_unsafe is not None else None,
            "centroid_bn_unsafe": self.centroid_bn_unsafe.tolist() if self.centroid_bn_unsafe is not None else None,
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved steering vector to {path}")
    
    def load_steering_vector(self, path: Path) -> None:
        """Load pre-computed steering vector."""
        with open(path, "r") as f:
            data = json.load(f)
        
        self.target_layer = data["target_layer"]
        self.steering_vector = torch.tensor(data["steering_vector"]) if data["steering_vector"] else None
        self.centroid_en_unsafe = torch.tensor(data["centroid_en_unsafe"]) if data["centroid_en_unsafe"] else None
        self.centroid_bn_unsafe = torch.tensor(data["centroid_bn_unsafe"]) if data["centroid_bn_unsafe"] else None
        
        logger.info(f"Loaded steering vector from {path}")
    
    def save_results(self, results: Dict, path: Path) -> None:
        """Save steering test results."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved steering results to {path}")
