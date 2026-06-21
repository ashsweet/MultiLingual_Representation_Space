"""Dataset loader for safety evaluation datasets with caching support."""
from typing import List, Dict, Optional
from pathlib import Path
import json

from datasets import load_dataset
from loguru import logger

from .config import Config


class DatasetLoader:
    """Loads safety evaluation datasets with local caching."""
    
    CACHE_DIR = Path("data/cache")
    CACHE_FILE = "bengali_prompts.json"
    
    def __init__(self, config: Config):
        self.config = config
        self.dataset = None
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    @property
    def cache_path(self) -> Path:
        return self.CACHE_DIR / self.CACHE_FILE
    
    def _load_from_cache(self) -> Optional[List[Dict]]:
        """Load cached prompts if available."""
        if self.cache_path.exists():
            logger.info(f"Loading from cache: {self.cache_path}")
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} prompts from cache")
            return data
        return None
    
    def _save_to_cache(self, pairs: List[Dict]) -> None:
        """Save prompts to cache."""
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        logger.info(f"Cached {len(pairs)} prompts to {self.cache_path}")
    
    def load(self) -> None:
        """Load dataset from HuggingFace."""
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        self.dataset = load_dataset(
            self.config.dataset_name, 
            split="validation",
            token=self.config.hf_token
        )
        logger.info(f"Dataset loaded: {len(self.dataset)} total samples")
    
    def get_parallel_prompts(
        self, 
        max_samples: Optional[int] = None,
        use_cache: bool = True,
        refresh_cache: bool = False
    ) -> List[Dict[str, str]]:
        """
        Extract English-Bengali prompt pairs.
        
        Args:
            max_samples: Limit number of samples
            use_cache: Use cached data if available
            refresh_cache: Force refresh cache from HuggingFace
        """
        # Try cache first (unless refresh requested)
        if use_cache and not refresh_cache:
            cached = self._load_from_cache()
            if cached:
                limit = max_samples or self.config.max_samples
                if limit and len(cached) > limit:
                    cached = cached[:limit]
                return cached
        
        # Load from HuggingFace
        if self.dataset is None:
            self.load()
        
        pairs = []
        
        # Filter for Bengali samples
        for idx, item in enumerate(self.dataset):
            if item.get("language") == "bn":
                pairs.append({
                    "id": idx,
                    "en": item["prompt_en"],
                    "bn": item["prompt_target"],
                    "category": ", ".join(item.get("tags", ["unknown"]))
                })
        
        logger.info(f"Found {len(pairs)} Bengali-English prompt pairs")
        
        # Cache all pairs (before limiting)
        if use_cache:
            self._save_to_cache(pairs)
        
        # Apply limit
        limit = max_samples or self.config.max_samples
        if limit and len(pairs) > limit:
            pairs = pairs[:limit]
            logger.info(f"Limited to {limit} samples")
        
        return pairs
    
    def clear_cache(self) -> None:
        """Clear the cached data."""
        if self.cache_path.exists():
            self.cache_path.unlink()
            logger.info("Cache cleared")
