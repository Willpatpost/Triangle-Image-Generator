from __future__ import annotations

import numpy as np

from core.config import Config
from core.shapes import Individual


def normalized_mse(reference: np.ndarray, rendered: np.ndarray) -> float:
    diff = reference.astype(np.float32) - rendered.astype(np.float32)
    mse = float(np.mean(diff ** 2))
    return mse / (255.0 ** 2)


def shape_penalty(num_shapes: int, config: Config) -> float:
    if config.nb_elements_max <= 0:
        return 0.0
    return config.shape_penalty_weight * (num_shapes / config.nb_elements_max)


def evaluate_fitness(
    individual: Individual,
    reference: np.ndarray,
    config: Config,
    background: int | tuple[int, int, int],
) -> float:
    from core.renderer import render_individual

    rendered = render_individual(individual, config, background)
    fitness = normalized_mse(reference, rendered) + shape_penalty(len(individual.triangles), config)
    individual.fitness = fitness
    return fitness


def evaluate_population(
    population: list[Individual],
    reference: np.ndarray,
    config: Config,
    background: int | tuple[int, int, int],
) -> list[tuple[int, float]]:
    for individual in population:
        if individual.fitness == float("inf"):
            evaluate_fitness(individual, reference, config, background)

    ranked = sorted(
        ((i, ind.fitness) for i, ind in enumerate(population)),
        key=lambda item: item[1],
    )
    return ranked


def _evaluate_individual_task(args: tuple) -> tuple[int, float]:
    index, individual, reference, config, background = args
    fitness = evaluate_fitness(individual, reference, config, background)
    return index, fitness


def evaluate_population_parallel(
    population: list[Individual],
    reference: np.ndarray,
    config: Config,
    background: int | tuple[int, int, int],
    max_workers: int,
) -> list[tuple[int, float]]:
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    pending = [
        (i, ind)
        for i, ind in enumerate(population)
        if ind.fitness == float("inf")
    ]
    if not pending:
        return evaluate_population(population, reference, config, background)

    workers = max_workers if max_workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    tasks = [
        (i, ind, reference, config, background)
        for i, ind in pending
    ]

    if workers == 1 or len(tasks) == 1:
        for i, ind in pending:
            evaluate_fitness(ind, reference, config, background)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_evaluate_individual_task, task) for task in tasks]
            for future in as_completed(futures):
                index, fitness = future.result()
                population[index].fitness = fitness

    ranked = sorted(
        ((i, ind.fitness) for i, ind in enumerate(population)),
        key=lambda item: item[1],
    )
    return ranked
