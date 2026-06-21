"""PyTorch forward hooks for capturing hidden states."""

from typing import Optional
import torch
from torch import Tensor


class HiddenStateHook:
    """Forward hook to capture hidden states from transformer layers.
    
    Uses attention-masked mean pooling to properly handle variable-length
    sequences and ignore padding tokens.
    """
    
    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.hidden_state: Optional[Tensor] = None
        self.attention_mask: Optional[Tensor] = None
        self.handle = None
    
    def set_attention_mask(self, attention_mask: Tensor):
        """Set the attention mask before forward pass.
        
        Args:
            attention_mask: Shape [batch, seq_len], 1 for real tokens, 0 for padding
        """
        self.attention_mask = attention_mask
    
    def __call__(self, module, input, output):
        """Hook function called during forward pass."""
        # output is typically (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        
        # Shape: [batch, seq_len, hidden_dim]
        if hidden.dim() == 3:
            # Use attention-masked mean pooling to ignore padding tokens
            pooled = self._masked_mean_pooling(hidden)
            self.hidden_state = pooled.detach().cpu()
        else:
            self.hidden_state = hidden.detach().cpu()
    
    def _masked_mean_pooling(self, hidden: Tensor) -> Tensor:
        """Compute mean pooling with attention mask to ignore padding.
        
        Args:
            hidden: Shape [batch, seq_len, hidden_dim]
        
        Returns:
            Pooled representation [batch, hidden_dim]
        """
        if self.attention_mask is None:
            # Fallback: use simple mean (not recommended)
            return hidden.mean(dim=1)
        
        # Expand attention mask to match hidden dims
        # attention_mask: [batch, seq_len] -> [batch, seq_len, 1]
        mask = self.attention_mask.unsqueeze(-1).to(hidden.device, dtype=hidden.dtype)
        
        # Mask and sum: only real tokens contribute
        # hidden * mask zeros out padding positions
        masked_hidden = hidden * mask
        sum_hidden = masked_hidden.sum(dim=1)  # [batch, hidden_dim]
        
        # Count real tokens (avoid division by zero)
        token_counts = mask.sum(dim=1).clamp(min=1e-9)  # [batch, 1]
        
        # Compute mean over real tokens only
        pooled = sum_hidden / token_counts
        
        return pooled
    
    def register(self, layer_module):
        """Register hook on a layer."""
        self.handle = layer_module.register_forward_hook(self)
    
    def remove(self):
        """Remove the hook."""
        if self.handle:
            self.handle.remove()
            self.handle = None
    
    def get_state(self) -> Optional[Tensor]:
        """Get captured hidden state and clear."""
        state = self.hidden_state
        self.hidden_state = None
        return state
    
    def clear_mask(self):
        """Clear the attention mask after use."""
        self.attention_mask = None
