"""Hidden State Analyzer Module - Tracks representation collapse across transformer layers."""

from .analyzer import HiddenStateAnalyzer
from .hooks import HiddenStateHook
from .types import HiddenStateCapture, SimilarityResult
from .failure_loader import (
    load_phase1_prompts,
    load_fragmentation_failures, 
    get_failure_ids, 
    FRAGMENTATION_FAILURE_IDS
)

__all__ = [
    "HiddenStateAnalyzer",
    "HiddenStateHook", 
    "HiddenStateCapture",
    "SimilarityResult",
    "load_phase1_prompts",
    "load_fragmentation_failures",
    "get_failure_ids",
    "FRAGMENTATION_FAILURE_IDS",
]
