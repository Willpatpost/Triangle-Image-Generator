"""Core library for triangle-based genetic image approximation."""

from core.config import Config

__all__ = ["Config"]


def run_evolution(*args, **kwargs):
    from core.pipeline import run_evolution as _run

    return _run(*args, **kwargs)
