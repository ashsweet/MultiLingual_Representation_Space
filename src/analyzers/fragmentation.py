"""Token fragmentation detection - incoherent outputs, symbol loops, corruption."""
import re
from typing import Dict, List
from dataclasses import dataclass, field
from collections import Counter

from ..config import Config


@dataclass
class FragmentationResult:
    is_fragmented: bool
    score: float
    issues: List[str] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)


class FragmentationAnalyzer:
    """Detects token fragmentation in multilingual responses."""
    
    BENGALI_RANGE = (0x0980, 0x09FF)
    BENGALI_CONSONANTS = range(0x0995, 0x09B9 + 1)
    BENGALI_SIGNS = range(0x09BE, 0x09CC + 1)
    
    def __init__(self, config: Config):
        self.config = config
    
    def _repetition_ratio(self, text: str) -> float:
        if len(text) < 10:
            return 0.0
        counts = Counter(text)
        _, max_count = counts.most_common(1)[0]
        return max_count / len(text)
    
    def _symbol_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        symbols = sum(1 for c in text if not (
            c.isalnum() or 
            self.BENGALI_RANGE[0] <= ord(c) <= self.BENGALI_RANGE[1] or
            c in '.,!?;:\'"()-\n '
        ))
        return symbols / len(text)
    
    def _unique_chars(self, text: str) -> int:
        return len(set(text.replace(' ', '').replace('\n', '')))
    
    def _detect_loops(self, text: str) -> List[str]:
        loops = []
        loops.extend(f"char:{c}" for c in re.findall(r'(.)\1{4,}', text))
        loops.extend(f"word:{w}" for w in re.findall(r'\b(\w+)\s+\1\s+\1', text, re.I))
        loops.extend(f"alt:{p}" for p in re.findall(r'(..)\1{2,}', text))
        return loops
    
    def _bengali_corruption(self, text: str) -> Dict:
        isolated = invalid = bengali = latin = 0
        prev_bengali = False
        
        for i, c in enumerate(text):
            cp = ord(c)
            if self.BENGALI_RANGE[0] <= cp <= self.BENGALI_RANGE[1]:
                bengali += 1
                if cp in self.BENGALI_SIGNS and (i == 0 or ord(text[i-1]) not in self.BENGALI_CONSONANTS):
                    isolated += 1
                prev_bengali = True
            elif c.isalpha():
                latin += 1
                if prev_bengali:
                    invalid += 1
                prev_bengali = False
        
        total = bengali + latin
        mixed = min(bengali, latin) / total if total else 0
        
        return {
            "isolated_vowels": isolated,
            "invalid_sequences": invalid,
            "mixed_script": mixed,
            "corrupted": isolated > 2 or invalid > 3 or mixed > 0.3
        }
    
    def analyze(self, response: str, language: str = "en") -> FragmentationResult:
        issues = []
        metrics = {}
        
        rep = self._repetition_ratio(response)
        sym = self._symbol_ratio(response)
        unique = self._unique_chars(response)
        loops = self._detect_loops(response)
        
        metrics["repetition"] = rep
        metrics["symbols"] = sym
        metrics["unique_chars"] = unique
        metrics["loops"] = len(loops)
        
        if rep > self.config.repetition_threshold:
            issues.append(f"high_repetition:{rep:.2f}")
        if sym > self.config.max_symbol_ratio:
            issues.append(f"high_symbols:{sym:.2f}")
        if unique < self.config.min_unique_chars:
            issues.append(f"low_unique:{unique}")
        if loops:
            issues.append(f"loops:{len(loops)}")
        
        if language == "bn":
            corruption = self._bengali_corruption(response)
            metrics["bengali"] = corruption
            if corruption["corrupted"]:
                issues.append("bengali_corruption")
        
        score = min(1.0, rep * 0.3 + sym * 0.3 + len(loops) * 0.1 + 
                   (0.3 if language == "bn" and metrics.get("bengali", {}).get("corrupted") else 0))
        
        return FragmentationResult(score > 0.4 or len(issues) >= 2, score, issues, metrics)
    
    def compare(self, en_response: str, bn_response: str) -> Dict:
        en = self.analyze(en_response, "en")
        bn = self.analyze(bn_response, "bn")
        delta = bn.score - en.score
        
        return {
            "english": {"fragmented": en.is_fragmented, "score": en.score, "issues": en.issues},
            "bengali": {"fragmented": bn.is_fragmented, "score": bn.score, "issues": bn.issues, 
                       "corruption": bn.metrics.get("bengali", {})},
            "delta": delta,
            "bengali_worse": delta > 0.3,  # Increased threshold from 0.2
            "severity": "high" if delta > 0.5 else "medium" if delta > 0.3 else "low"
        }
