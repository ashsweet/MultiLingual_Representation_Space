"""Load prompts from Phase 1 results - both successes (safe) and failures (unsafe)."""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from loguru import logger


# IDs of prompts that had token_fragmentation issues (35 total) - these are UNSAFE/FAILURES
# Used as fallback if 35_failure_cohort.json doesn't exist
FRAGMENTATION_FAILURE_IDS = [
    2, 5, 7, 15, 20, 24, 27, 37, 46, 50, 58, 61, 70, 72, 75,
    85, 89, 90, 103, 104, 105, 107, 109, 111, 112, 114, 115,
    122, 124, 129, 131, 133, 138, 139, 143
]


def _find_responses_path() -> Optional[Path]:
    """Find the most recent responses.json from Phase 1."""
    results_dir = Path("results")
    
    # First check in new Phase1 directory structure
    phase1_dir = results_dir / "Phase1_Behavioral_Profiling"
    if phase1_dir.exists():
        run_dirs = sorted(phase1_dir.glob("run_*"), reverse=True)
        for d in run_dirs:
            responses_path = d / "responses.json"
            if responses_path.exists():
                return responses_path
    
    # Fallback to old structure for backward compatibility
    if results_dir.exists():
        run_dirs = sorted(results_dir.glob("run_*"), reverse=True)
        for d in run_dirs:
            if "Phase" not in str(d):  # Skip new phase directories
                responses_path = d / "responses.json"
                if responses_path.exists():
                    return responses_path
    return None


def load_phase1_prompts(
    responses_path: Optional[Path] = None,
    max_prompts: int = 35
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load SAFE (successful) and UNSAFE (failure) prompts from Phase 1.
    
    First tries to load from data/150_baseline_prompts.json and data/35_failure_cohort.json.
    Falls back to responses.json if those don't exist.
    
    Safe = prompts where model behaved correctly (no fragmentation)
    Unsafe = prompts that had token_fragmentation issues (35 failures)
    
    Args:
        responses_path: Path to responses.json from Phase 1 (fallback)
        max_prompts: Maximum number of prompts per category
    
    Returns:
        Tuple of (safe_prompts, unsafe_prompts)
        Each is List of {'id': int, 'en': str, 'bn': str, 'category': str}
    """
    # Try to load from data/ files first
    baseline_path = Path("data/150_baseline_prompts.json")
    cohort_path = Path("data/35_failure_cohort.json")
    
    if baseline_path.exists() and cohort_path.exists():
        logger.info("Loading from data/ files...")
        
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        with open(cohort_path, "r", encoding="utf-8") as f:
            failure_cohort = json.load(f)
        
        # Get failure IDs
        failure_ids = {p["id"] for p in failure_cohort}
        
        # Split baseline into safe and unsafe
        safe_prompts = [p for p in baseline if p["id"] not in failure_ids]
        unsafe_prompts = failure_cohort
        
        logger.info(f"Loaded {len(safe_prompts)} SAFE prompts from baseline")
        logger.info(f"Loaded {len(unsafe_prompts)} UNSAFE prompts from failure cohort")
        
    else:
        # Fallback to responses.json
        logger.info("data/ files not found, falling back to responses.json...")
        
        if responses_path is None:
            responses_path = _find_responses_path()
        
        if responses_path is None or not responses_path.exists():
            raise FileNotFoundError(
                f"responses.json not found. Run Phase 1 analysis first with: python main.py"
            )
        
        logger.info(f"Loading Phase 1 prompts from: {responses_path}")
        
        with open(responses_path, "r", encoding="utf-8") as f:
            responses = json.load(f)
        
        safe_prompts = []
        unsafe_prompts = []
        
        for item in responses:
            prompt_data = {
                "id": item["id"],
                "en": item["en_prompt"],
                "bn": item["bn_prompt"],
                "category": item.get("category", "unknown")
            }
            
            if item["id"] in FRAGMENTATION_FAILURE_IDS:
                unsafe_prompts.append(prompt_data)
            else:
                safe_prompts.append(prompt_data)
        
        logger.info(f"Loaded {len(safe_prompts)} SAFE prompts (successful)")
        logger.info(f"Loaded {len(unsafe_prompts)} UNSAFE prompts (fragmentation failures)")
    
    # Limit to max_prompts each
    if len(safe_prompts) > max_prompts:
        safe_prompts = safe_prompts[:max_prompts]
    if len(unsafe_prompts) > max_prompts:
        unsafe_prompts = unsafe_prompts[:max_prompts]
    
    logger.info(f"Using {len(safe_prompts)} safe and {len(unsafe_prompts)} unsafe prompts")
    
    return safe_prompts, unsafe_prompts


def load_fragmentation_failures(
    responses_path: Optional[Path] = None,
    max_prompts: int = 35
) -> List[Dict[str, str]]:
    """
    Load prompts that had token fragmentation issues from Phase 1.
    (Backward compatibility wrapper)
    """
    _, unsafe_prompts = load_phase1_prompts(responses_path, max_prompts)
    return unsafe_prompts


def get_failure_ids() -> List[int]:
    """Return the list of fragmentation failure IDs."""
    return FRAGMENTATION_FAILURE_IDS.copy()
