from __future__ import annotations

import logging
import os
import time
from dataclasses import replace

import numpy as np

from core.config import ColorMode, Config
from core.fitness import evaluate_population_parallel
from core.genetic import create_initial_population, hill_climb_elite, next_generation
from core.io import (
    create_gif,
    load_reference_image,
    load_state,
    save_comparison,
    save_image,
    save_state,
)
from core.paths import (
    COMPARISON_IMAGE_NAME,
    EVOLUTION_GIF_NAME,
    FINAL_IMAGE_NAME,
    STATE_FILE_NAME,
    iteration_frame_path,
    output_artifact,
)
from core.renderer import render_individual


RUNTIME_CONFIG_FIELDS = {
    "save_directory",
    "save_comparison",
    "gif_frame_duration_ms",
    "enable_logging",
    "log_interval_sec",
    "log_level",
    "max_workers",
    "use_compositing_cache",
}


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
) -> tuple[np.ndarray, float]:
    if seed is not None:
        import random

        random.seed(seed)
        np.random.seed(seed)

    os.makedirs(config.save_directory, exist_ok=True)

    start_iteration = 0
    resume_individual = None
    if resume_path and os.path.isfile(resume_path):
        start_iteration, resume_fitness, resume_individual, loaded_config, loaded_bg = load_state(resume_path)
        runtime_overrides = {
            field: getattr(config, field)
            for field in RUNTIME_CONFIG_FIELDS
        }
        config = replace(loaded_config, **runtime_overrides)
        logging.info("Resumed from iteration %d (fitness %.6f)", start_iteration, resume_fitness)
        start_iteration += 1

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

    population = create_initial_population(width, height, config.mode, config, resume_individual)
    max_workers = config.max_workers

    ranked = evaluate_population_parallel(population, reference, config, background, max_workers)
    best_index, best_fitness = ranked[0]
    last_saved_fitness = float("inf")
    last_log_time = time.time()
    stagnation = 0
    mutation_rate = config.mutation_rate

    global_best = population[best_index].copy()
    global_best.fitness = best_fitness

    for iteration in range(start_iteration, iterations):
        crossover_step = iteration % 2 == 0
        if stagnation > config.stagnation_threshold // 4:
            mutation_rate = min(config.mutation_rate * config.adaptive_mutation_boost, 3.0)
        else:
            mutation_rate = config.mutation_rate

        population = next_generation(
            population,
            ranked,
            config,
            mutation_rate,
            crossover_generation=crossover_step,
        )

        if config.hill_climb_interval > 0 and iteration % config.hill_climb_interval == 0:
            seed_elite = population[0].copy()
            for _ in range(config.hill_climb_attempts):
                population.append(hill_climb_elite(seed_elite, config))

        for individual in population:
            individual.fitness = float("inf")

        ranked = evaluate_population_parallel(population, reference, config, background, max_workers)

        if len(population) > config.pop_size:
            population = [population[i] for i, _ in ranked[: config.pop_size]]
            ranked = sorted(
                ((i, individual.fitness) for i, individual in enumerate(population)),
                key=lambda item: item[1],
            )

        best_index, best_fitness = ranked[0]

        if best_fitness < global_best.fitness:
            global_best = population[best_index].copy()
            global_best.fitness = best_fitness
            stagnation = 0
            logging.info("New best fitness %.6f at iteration %d", best_fitness, iteration)
        else:
            stagnation += 1

        if time.time() - last_log_time >= config.log_interval_sec:
            logging.info(
                "Iteration %d | triangles %d | fitness %.6f | mutation %.2f",
                iteration,
                len(population[best_index].triangles),
                best_fitness,
                mutation_rate,
            )
            last_log_time = time.time()

        if best_fitness < last_saved_fitness:
            frame_path = iteration_frame_path(config.save_directory, iteration)
            rendered = render_individual(global_best, config, background)
            save_image(rendered, str(frame_path), config.mode)
            save_state(
                str(output_artifact(config.save_directory, STATE_FILE_NAME)),
                iteration,
                global_best,
                best_fitness,
                config,
                background,
                source_image=os.path.abspath(image_path),
                downsample=downsample,
            )
            last_saved_fitness = best_fitness

        if best_fitness <= config.fitness_goal:
            logging.info("Fitness goal reached at iteration %d", iteration)
            break

        if stagnation >= config.stagnation_threshold:
            logging.info("Stopping after %d iterations without improvement", stagnation)
            break

    final = render_individual(global_best, config, background)
    final_path = output_artifact(config.save_directory, FINAL_IMAGE_NAME)
    save_image(final, str(final_path), config.mode)

    if config.save_comparison:
        comparison_path = output_artifact(config.save_directory, COMPARISON_IMAGE_NAME)
        save_comparison(reference, final, str(comparison_path), config.mode)

    create_gif(config.save_directory, EVOLUTION_GIF_NAME, config.gif_frame_duration_ms)
    logging.info("Evolution complete. Best fitness: %.6f", global_best.fitness)
    return final, global_best.fitness
