from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from core.config import Config, RUNTIME_CONFIG_FIELDS
from core.fitness import evaluate_population_parallel
from core.genetic import create_initial_population, hill_climb_elite, next_generation
from core.guidance import EvolutionGuide
from core.io import (
    file_sha256,
    load_reference_image,
    read_state,
    save_image,
)
from core.renderer import render_individual, render_individual_float


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
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if downsample < 1:
        raise ValueError("downsample must be at least 1")
    config.validate()

    if seed is not None:
        import random

        random.seed(seed)
        np.random.seed(seed)

    start_iteration = 0
    resume_individual = None
    if resume_path and os.path.isfile(resume_path):
        state = read_state(resume_path)
        if state.source_sha256 and file_sha256(image_path) != state.source_sha256:
            raise ValueError("The reference image does not match the image saved with this session.")
        start_iteration = state.iteration + (1 if state.version == 1 else 0)
        resume_fitness = state.best_fitness
        resume_individual = state.best
        loaded_config = state.config
        loaded_bg = state.background
        runtime_overrides = {
            field: getattr(config, field)
            for field in RUNTIME_CONFIG_FIELDS
        }
        config = replace(loaded_config, **runtime_overrides)
        config.validate()
        logging.info("Resumed from iteration %d (fitness %.6f)", start_iteration, resume_fitness)

    reference, background = load_reference_image(
        image_path,
        config.mode,
        downsample,
        config.gaussian_blur_sigma,
    )
    if resume_individual is not None:
        height, width = reference.shape[:2]
        if resume_individual.mode != config.mode:
            raise ValueError(
                f"Resume mode {resume_individual.mode!r} does not match configured mode {config.mode!r}."
            )
        if (resume_individual.width, resume_individual.height) != (width, height):
            raise ValueError(
                "Resume state dimensions "
                f"{resume_individual.width}x{resume_individual.height} do not match "
                f"the current reference {width}x{height}. Use the same image and downsample."
            )
        background = loaded_bg

    config = replace(config, background=background)
    height, width = reference.shape[:2]
    logging.info("Reference image: %dx%d (%s)", width, height, config.mode)
    logging.info("Background color: %s", background)

    initial_guide = EvolutionGuide.from_background(reference, background)
    population = create_initial_population(
        width,
        height,
        config.mode,
        config,
        resume_individual,
        initial_guide,
    )
    max_workers = config.max_workers

    ranked = evaluate_population_parallel(population, reference, config, background, max_workers)
    best_index, best_fitness = ranked[0]
    last_log_time = time.time()
    stagnation = 0
    mutation_rate = config.mutation_rate

    global_best = population[best_index].copy()
    global_best.fitness = best_fitness

    for iteration in range(start_iteration, iterations):
        if stagnation > config.stagnation_threshold // 4:
            mutation_rate = min(config.mutation_rate * config.adaptive_mutation_boost, 3.0)
        else:
            mutation_rate = config.mutation_rate

        baseline = render_individual_float(global_best, config, background)
        guide = EvolutionGuide(reference, baseline)
        population = next_generation(
            population,
            ranked,
            config,
            mutation_rate,
            crossover_generation=True,
            guide=guide,
        )

        if (
            config.hill_climb_interval > 0
            and iteration > start_iteration
            and iteration % config.hill_climb_interval == 0
        ):
            seed_elite = population[0].copy()
            for _ in range(config.hill_climb_attempts):
                population.append(hill_climb_elite(seed_elite, config, guide=guide))

        ranked = evaluate_population_parallel(population, reference, config, background, max_workers)

        if len(population) > config.pop_size:
            population = [population[i] for i, _ in ranked[: config.pop_size]]
            ranked = sorted(
                ((i, individual.fitness) for i, individual in enumerate(population)),
                key=lambda item: item[1],
            )

        best_index, current_best_fitness = ranked[0]

        if current_best_fitness < global_best.fitness:
            global_best = population[best_index].copy()
            global_best.fitness = current_best_fitness
            stagnation = 0
            logging.info("New best fitness %.6f at iteration %d", current_best_fitness, iteration)
        else:
            stagnation += 1

        if time.time() - last_log_time >= config.log_interval_sec:
            logging.info(
                "Iteration %d | shapes %d | current %.6f | best %.6f | mutation %.2f",
                iteration,
                len(population[best_index].shapes),
                current_best_fitness,
                global_best.fitness,
                mutation_rate,
            )
            last_log_time = time.time()

        if global_best.fitness <= config.fitness_goal:
            logging.info("Fitness goal reached at iteration %d", iteration)
            break

        if stagnation >= config.stagnation_threshold:
            logging.info("Stopping after %d iterations without improvement", stagnation)
            break

    final = render_individual(global_best, config, background)
    if save_path is not None:
        save_image(final, save_path, config.mode)
    logging.info("Evolution complete. Best fitness: %.6f", global_best.fitness)
    return final, global_best.fitness


def run_progressive_evolution(
    image_path: str,
    config: Config,
    stages: Sequence[tuple[int, int]],
    *,
    seed: int | None = None,
    save_path: str | None = None,
) -> tuple[np.ndarray, float]:
    """Evolve from coarse to fine using ``(downsample, iterations)`` stages."""
    from core.evolver import EvolutionSession

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
        save_image(final, save_path, config.mode)
    return final, session.global_best.fitness
