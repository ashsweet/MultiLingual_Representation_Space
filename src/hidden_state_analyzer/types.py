"""Data types for hidden state analysis."""

from dataclasses import dataclass, field
from typing import Dict
from torch import Tensor


@dataclass
class HiddenStateCapture:
    """Captured hidden states for a single prompt."""
    prompt_id: str
    language: str
    prompt_type: str  # 'safe' or 'unsafe'
    prompt_text: str
    layer_states: Dict[int, Tensor] = field(default_factory=dict)  # layer_idx -> hidden state


@dataclass 
class SimilarityResult:
    """Cosine similarity results between prompt pairs."""
    layer_idx: int
    en_safe_vs_en_unsafe: float
    bn_safe_vs_bn_unsafe: float
    en_safe_vs_bn_safe: float
    en_unsafe_vs_bn_unsafe: float
    bn_collapse_score: float  # How much BN safe/unsafe are similar (higher = more collapse)
