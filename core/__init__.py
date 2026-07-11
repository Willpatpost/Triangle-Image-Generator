"""Core library for triangle-based genetic image approximation."""

from core.config import Config
from core.evolver import EvolutionSession, EvolutionSnapshot

__all__ = ["Config", "EvolutionSession", "EvolutionSnapshot"]


def run_evolution(*args, **kwargs):
    from core.pipeline import run_evolution as _run

    return _run(*args, **kwargs)
