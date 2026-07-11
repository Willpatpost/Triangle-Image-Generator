from __future__ import annotations

import numpy as np

from core.config import Config
from core.shapes import Individual


class _EliteCacheTracker:
    def __init__(self, population: list[Individual], elite_count: int) -> None:
        self.population = population
        self.elite_count = min(max(1, elite_count), len(population))
        finite = [
            index
            for index, individual in enumerate(population)
            if individual.fitness != float("inf")
        ]
        finite.sort(key=lambda index: population[index].fitness)
        self.retained = set(finite[: self.elite_count])
        for index in finite[self.elite_count :]:
            population[index].clear_caches()

    def consider(self, index: int) -> None:
        self.retained.add(index)
        if len(self.retained) <= self.elite_count:
            return
        worst = max(self.retained, key=lambda item: self.population[item].fitness)
        self.retained.remove(worst)
        self.population[worst].clear_caches()

    def finalize(self, ranked: list[tuple[int, float]]) -> None:
        keep = {index for index, _ in ranked[: self.elite_count]}
        for index, individual in enumerate(self.population):
            if index not in keep:
                individual.clear_caches()
        self.retained = keep


def squared_error_sum(reference: np.ndarray, rendered: np.ndarray) -> float:
    if reference.shape != rendered.shape:
        raise ValueError("reference and rendered images must have matching shapes")
    diff = np.subtract(reference, rendered, dtype=np.float32)
    np.square(diff, out=diff)
    return float(np.sum(diff, dtype=np.float64))


def normalized_mse(reference: np.ndarray, rendered: np.ndarray) -> float:
    return squared_error_sum(reference, rendered) / (reference.size * 255.0**2)


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
    from core.renderer import render_individual_for_scoring

    result = render_individual_for_scoring(individual, config, background)
    if (
        result.incremental
        and result.dirty_bbox is not None
        and result.previous_image is not None
        and result.previous_error_sum is not None
    ):
        min_x, min_y, max_x, max_y = result.dirty_bbox
        reference_region = reference[min_y : max_y + 1, min_x : max_x + 1]
        old_region = result.previous_image[min_y : max_y + 1, min_x : max_x + 1]
        new_region = result.image[min_y : max_y + 1, min_x : max_x + 1]
        error_sum = (
            result.previous_error_sum
            - squared_error_sum(reference_region, old_region)
            + squared_error_sum(reference_region, new_region)
        )
        error_sum = max(0.0, error_sum)
    else:
        error_sum = squared_error_sum(reference, result.image)

    individual._pixel_error_sum = error_sum
    normalized_error = error_sum / (reference.size * 255.0**2)
    fitness = normalized_error + shape_penalty(len(individual.triangles), config)
    individual.fitness = fitness
    return fitness


def evaluate_population(
    population: list[Individual],
    reference: np.ndarray,
    config: Config,
    background: int | tuple[int, int, int],
) -> list[tuple[int, float]]:
    tracker = _EliteCacheTracker(population, config.nb_elite)
    for index, individual in enumerate(population):
        if individual.fitness == float("inf"):
            evaluate_fitness(individual, reference, config, background)
            tracker.consider(index)

    ranked = sorted(
        ((i, ind.fitness) for i, ind in enumerate(population)),
        key=lambda item: item[1],
    )
    tracker.finalize(ranked)
    return ranked


def evaluate_population_parallel(
    population: list[Individual],
    reference: np.ndarray,
    config: Config,
    background: int | tuple[int, int, int],
    max_workers: int,
) -> list[tuple[int, float]]:
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    pending = [
        (i, ind)
        for i, ind in enumerate(population)
        if ind.fitness == float("inf")
    ]
    if not pending:
        return evaluate_population(population, reference, config, background)

    tracker = _EliteCacheTracker(population, config.nb_elite)
    workers = max_workers if max_workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    workers = min(workers, len(pending), 4)
    from core.acceleration import resolve_renderer_backend

    backend = resolve_renderer_backend(config.renderer_backend, reference.size, config)
    if backend == "cuda":
        workers = 1
    work_size = reference.size
    parallel_threshold = config.parallel_min_pixels
    if config.shape_mode == "voronoi":
        parallel_threshold = min(parallel_threshold, 131_072)
    if workers == 1 or len(pending) == 1 or work_size < parallel_threshold:
        for i, ind in pending:
            evaluate_fitness(ind, reference, config, background)
            tracker.consider(i)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="shape-fitness") as executor:
            futures = {
                executor.submit(evaluate_fitness, ind, reference, config, background): index
                for index, ind in pending
            }
            for future in as_completed(futures):
                future.result()
                tracker.consider(futures[future])

    ranked = sorted(
        ((i, ind.fitness) for i, ind in enumerate(population)),
        key=lambda item: item[1],
    )
    tracker.finalize(ranked)
    return ranked
