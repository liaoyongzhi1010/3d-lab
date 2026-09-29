"""Central candidate registry: build any registered method by name + cfg."""

from __future__ import annotations

from typing import Dict, Type

from .reconstruction_interface import ReconstructionModel

_REGISTRY: Dict[str, Type[ReconstructionModel]] = {}


def register(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls

    return decorator


def build_candidate(name: str, cfg: dict) -> ReconstructionModel:
    if name not in _REGISTRY:
        raise ValueError(f"unknown candidate '{name}'; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](cfg)


def registered_names():
    return sorted(_REGISTRY.keys())


from .candidate_regression_residual import (
    RegressionResidualControl,
    ConfidenceRoutedResidual,
)
from .candidate_latent_gaussian import LatentGaussianGeneration
from .candidate_teacher_distill import TeacherDistillation
from .candidate_shared3d_latent import Shared3DLatentRefinement
from .candidate_multihypothesis import MultiHypothesisSelector

register("C0")(RegressionResidualControl)
register("C1")(ConfidenceRoutedResidual)
register("C2")(LatentGaussianGeneration)
register("C3")(TeacherDistillation)
register("C4")(Shared3DLatentRefinement)
register("C5")(MultiHypothesisSelector)
