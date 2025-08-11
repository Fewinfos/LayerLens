"""
Core components for the LayerLens explanation engine.
"""

from layerlens.core.surrogate_builder import SurrogateBuilder
from layerlens.core.layer_extractor import LayerExtractor
from layerlens.core.stitching_engine import StitchingEngine
from layerlens.core.model_hooks import register_hooks, remove_hooks
from layerlens.core.fidelity_metrics import calculate_fidelity, FidelityScore

__all__ = [
    'SurrogateBuilder',
    'LayerExtractor',
    'StitchingEngine',
    'register_hooks',
    'remove_hooks',
    'calculate_fidelity',
    'FidelityScore'
]
