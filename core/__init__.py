"""Core library for genetic image approximation with primitive shapes."""

from core.acceleration import (
    AccelerationStatus,
    AcceleratorUnavailable,
    acceleration_status,
    available_renderer_backends,
)
from core.config import Config
from core.evolver import EvolutionSession, EvolutionSnapshot
from core.io import EvolutionState

__all__ = [
    "Config",
    "EvolutionSession",
    "EvolutionSnapshot",
    "EvolutionState",
    "AccelerationStatus",
    "AcceleratorUnavailable",
    "acceleration_status",
    "available_renderer_backends",
    "run_evolution",
    "run_progressive_evolution",
]


def run_evolution(*args, **kwargs):
    from core.pipeline import run_evolution as _run

    return _run(*args, **kwargs)


def run_progressive_evolution(*args, **kwargs):
    from core.pipeline import run_progressive_evolution as _run

    return _run(*args, **kwargs)
