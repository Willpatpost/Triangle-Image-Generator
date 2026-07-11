"""Core library for triangle-based genetic image approximation."""

from core.acceleration import (
    AccelerationStatus,
    AcceleratorUnavailable,
    acceleration_status,
    available_renderer_backends,
)
from core.config import Config
from core.evolver import EvolutionSession, EvolutionSnapshot

__all__ = [
    "Config",
    "EvolutionSession",
    "EvolutionSnapshot",
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
