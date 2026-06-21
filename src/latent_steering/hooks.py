"""Steering hooks for modifying hidden states during forward pass."""

import torch
from torch import Tensor
from typing import Optional, Tuple, Union


class SteeringHook:
    """
    Forward hook that adds a steering vector to hidden states.
    
    Formula: h_corrected = h_bengali + alpha * v_steer
    
    Where v_steer = centroid_en_unsafe - centroid_bn_unsafe
    This pushes Bengali representations toward the English safety anchor.
    """
    
    def __init__(self, layer_idx: int, steering_vector: Tensor, alpha: float = 1.0):
        """
        Args:
            layer_idx: Which layer this hook is attached to
            steering_vector: Direction to push hidden states [hidden_dim]
            alpha: Scaling factor for steering strength (default 1.0)
        """
        self.layer_idx = layer_idx
        self.steering_vector = steering_vector  # [hidden_dim]
        self.alpha = alpha
        self.handle = None
        self.enabled = True
    
    def __call__(
        self, 
        module, 
        input: Tuple[Tensor, ...], 
        output: Union[Tensor, Tuple[Tensor, ...]]
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Hook called during forward pass. Modifies hidden states in-place.
        
        Args:
            module: The layer module
            input: Input tensors to the layer
            output: Output from the layer (hidden_states, ...)
        
        Returns:
            Modified output with steered hidden states
        """
        if not self.enabled:
            return output
        
        # Handle tuple output (hidden_states, attention, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None
        
        # Apply steering: h_corrected = h + alpha * v_steer
        # hidden_states shape: [batch, seq_len, hidden_dim]
        # steering_vector shape: [hidden_dim]
        
        # Move steering vector to same device and dtype
        v_steer = self.steering_vector.to(
            device=hidden_states.device, 
            dtype=hidden_states.dtype
        )
        
        # Add steering vector to ALL token positions
        # Broadcasting: [batch, seq_len, hidden_dim] + [hidden_dim]
        steered = hidden_states + self.alpha * v_steer
        
        # Return with same structure as input
        if rest is not None:
            return (steered,) + rest
        return steered
    
    def register(self, layer_module) -> None:
        """Register this hook on a layer module."""
        self.handle = layer_module.register_forward_hook(self)
    
    def remove(self) -> None:
        """Remove the hook."""
        if self.handle:
            self.handle.remove()
            self.handle = None
    
    def set_alpha(self, alpha: float) -> None:
        """Adjust steering strength."""
        self.alpha = alpha
    
    def enable(self) -> None:
        """Enable steering."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable steering (hook still attached but does nothing)."""
        self.enabled = False
