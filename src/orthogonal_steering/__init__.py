"""
Orthogonal Projection Steering Module - Phase 4

Corrects Bengali safety drift using orthogonally-projected steering vectors
that decouple safety correction from language identity.

Two Solutions Implemented:

Solution A - Cross-lingual Orthogonal Projection:
    The raw steering vector (EN_unsafe - BN_unsafe) causes "Language Bleed"
    where steered responses switch to English. By projecting out the language axis,
    we create a clean safety vector that corrects safety without affecting language.
    
    Formulas:
        v_lang = c_BN_safe - c_EN_safe  (language axis)
        v_steer_raw = c_EN_unsafe - c_BN_unsafe  (raw safety)
        v_steer_clean = v_steer_raw - proj(v_steer_raw, v_lang)  (orthogonal)
        v_steer_scaled = v_steer_clean * (||v_steer_raw|| / ||v_steer_clean||)  (magnitude-preserved)

Solution B - Monolingual Safety Anchoring:
    Extracts the safety refusal direction purely within the stable English manifold,
    bypassing the cross-lingual translation vector completely.
    
    Formulas:
        v_safety_mono = c_EN_safe - c_EN_unsafe  (English safety direction)
        v_safety_mono_ortho = v_safety_mono - proj(v_safety_mono, v_lang)  (orthogonalized)
        v_safety_mono_scaled = v_safety_mono_ortho * (||v_safety_mono|| / ||v_safety_mono_ortho||)

The Collinearity Problem:
    At Layer 13, cos(v_steer_raw, v_lang) ~ -0.998 (almost anti-parallel).
    Orthogonal projection removes ~99% of the vector's magnitude.
    Magnitude-preserving normalization restores steering strength.

Modes available in apply_steering():
    - "raw": Original cross-lingual steering (causes language bleed)
    - "clean": Orthogonally projected (weak)
    - "scaled": Magnitude-preserved orthogonal (Solution A recommended)
    - "mono": Monolingual safety (c_EN_safe - c_EN_unsafe)
    - "mono_ortho": Orthogonalized monolingual
    - "mono_scaled": Magnitude-preserved monolingual (Solution B recommended)
"""

from .orthogonal_steering import OrthogonalSteering
from .visualizer import Phase4Visualizer, generate_phase4_visualizations

__all__ = [
    "OrthogonalSteering",
    "Phase4Visualizer",
    "generate_phase4_visualizations",
]
