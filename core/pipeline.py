from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from core.config import Config, RUNTIME_CONFIG_FIELDS
from core.evolver import EvolutionSession
from core.io import save_image


def setup_logging(level: str, enabled: bool) -> None:
    if not enabled:
        logging.basicConfig(level=logging.CRITICAL)
        return
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_evolution(
    image_path: str,
    config: Config,
    *,
    iterations: int = 30_000,
    downsample: int = 1,
    resume_path: str | None = None,
    seed: int | None = None,
    save_path: str | None = None,
) -> tuple[np.ndarray, float]:
    """Run a full evolution, optionally resuming a saved session.

    When ``resume_path`` is given, the saved session supplies the evolutionary
    configuration and RNG stream, so ``seed`` is ignored and ``downsample``
    must match the working resolution the session was saved with.
    """
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if downsample < 1:
        raise ValueError("downsample must be at least 1")
    config.validate()

    if resume_path is not None:
        if not Path(resume_path).is_file():
            raise FileNotFoundError(f"Resume session not found: {resume_path}")
        session = EvolutionSession.from_state(
            resume_path,
            image_path=image_path,
            downsample=downsample,
            runtime_overrides={field: getattr(config, field) for field in RUNTIME_CONFIG_FIELDS},
        )
        logging.info(
            "Resumed from iteration %d (fitness %.6f)",
            session.iteration,
            session.global_best.fitness,
        )
    else:
        session = EvolutionSession(image_path, config, downsample=downsample, seed=seed)

    height, width = session.reference.shape[:2]
    logging.info("Reference image: %dx%d (%s)", width, height, session.config.mode)
    logging.info("Background color: %s", session.background)

    last_log_time = time.time()
    while session.iteration < iterations:
        snapshot = session.step()
        if snapshot.improved:
            logging.info(
                "New best fitness %.6f at iteration %d",
                snapshot.best_fitness,
                session.iteration,
            )
        if time.time() - last_log_time >= session.config.log_interval_sec:
            logging.info(
                "Iteration %d | shapes %d | current %.6f | best %.6f | mutation %.2f",
                session.iteration,
                snapshot.shape_count,
                snapshot.current_fitness,
                snapshot.best_fitness,
                session.mutation_rate,
            )
            last_log_time = time.time()
        if session.global_best.fitness <= session.config.fitness_goal:
            logging.info("Fitness goal reached at iteration %d", session.iteration)
            break
        if session.stagnation >= session.config.stagnation_threshold:
            logging.info("Stopping after %d iterations without improvement", session.stagnation)
            break

    final = session.render_best()
    if save_path is not None:
        save_image(final, save_path, session.config.mode)
    logging.info("Evolution complete. Best fitness: %.6f", session.global_best.fitness)
    return final, session.global_best.fitness


def run_progressive_evolution(
    image_path: str,
    config: Config,
    stages: Sequence[tuple[int, int]],
    *,
    seed: int | None = None,
    save_path: str | None = None,
) -> tuple[np.ndarray, float]:
    """Evolve from coarse to fine using ``(downsample, iterations)`` stages."""
    if not stages:
        raise ValueError("at least one progressive stage is required")
    previous_downsample: int | None = None
    for downsample, iterations in stages:
        if downsample < 1 or iterations < 1:
            raise ValueError("each stage needs a positive downsample and iteration count")
        if previous_downsample is not None and downsample >= previous_downsample:
            raise ValueError("progressive downsample values must strictly decrease")
        previous_downsample = downsample

    first_downsample, _ = stages[0]
    session = EvolutionSession(image_path, config, downsample=first_downsample, seed=seed)
    for stage_index, (downsample, iterations) in enumerate(stages):
        if stage_index > 0:
            session.refine_resolution(downsample)
        for _ in range(iterations):
            session.step()
            if (
                session.global_best.fitness <= session.config.fitness_goal
                or session.stagnation >= session.config.stagnation_threshold
            ):
                break

    final = session.render_best()
    if save_path is not None:
        save_image(final, save_path, session.config.mode)
    return final, session.global_best.fitness
