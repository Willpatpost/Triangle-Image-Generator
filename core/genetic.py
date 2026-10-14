from __future__ import annotations

import random

from core.config import ColorMode, Config
from core.guidance import EvolutionGuide
from core.shapes import Individual


def create_initial_population(
    width: int,
    height: int,
    mode: ColorMode,
    config: Config,
    seed: Individual | None = None,
    guide: EvolutionGuide | None = None,
    *,
    rng: random.Random,
) -> list[Individual]:
    population: list[Individual] = []
    if seed is not None:
        seeded = seed.copy()
        seeded.fitness = seed.fitness if seed._pixel_error_sum is not None else float("inf")
        population.append(seeded)
    while len(population) < config.pop_size:
        population.append(
            Individual.random(
                width,
                height,
                mode,
                config,
                config.nb_elements_initial,
                guide,
                rng=rng,
            )
        )
    return population


def elite_candidate_batch(
    best: Individual,
    config: Config,
    mutation_rate: float,
    guide: EvolutionGuide | None = None,
    *,
    rng: random.Random,
    count: int,
) -> list[Individual]:
    """Fork-and-mutate candidates of the current best individual."""
    candidates = []
    for _ in range(count):
        candidate = best.fork()
        candidate.mutate(config, rate=mutation_rate * config.elite_candidate_rate, guide=guide, rng=rng)
        candidate.fitness = float("inf")
        candidates.append(candidate)
    return candidates


def hill_climb_elite(
    elite: Individual,
    config: Config,
    rate: float = 0.5,
    guide: EvolutionGuide | None = None,
    *,
    rng: random.Random,
) -> Individual:
    candidate = elite.fork()
    candidate.mutate(config, rate=config.mutation_rate * rate, guide=guide, rng=rng)
    candidate.fitness = float("inf")
    return candidate


def _select_elite(
    population: list[Individual],
    ranked: list[tuple[int, float]],
    elite_count: int,
    *,
    rng: random.Random,
) -> Individual:
    # Squaring biases selection toward the best ranks while preserving diversity.
    rank = min(elite_count - 1, int(rng.random() ** 2 * elite_count))
    return population[ranked[rank][0]]


def next_generation(
    population: list[Individual],
    ranked: list[tuple[int, float]],
    config: Config,
    mutation_rate: float,
    crossover_generation: bool,
    guide: EvolutionGuide | None = None,
    *,
    rng: random.Random,
) -> list[Individual]:
    elite_count = min(config.nb_elite, len(ranked))
    elite_sources = [population[ranked[index][0]] for index in range(elite_count)]
    offspring = [elite.fork() for elite in elite_sources]

    crossover_weight = config.prob_uniform_crossover + config.prob_blend_crossover
    while len(offspring) < config.pop_size:
        parent_a = _select_elite(population, ranked, elite_count, rng=rng)
        use_crossover = (
            crossover_generation
            and elite_count > 1
            and crossover_weight > 0
            and rng.random() < config.crossover_rate
        )
        if use_crossover:
            parent_b = _select_elite(population, ranked, elite_count, rng=rng)
            uniform_probability = config.prob_uniform_crossover / crossover_weight
            if rng.random() < uniform_probability:
                child = Individual.crossover_uniform(parent_a, parent_b, config, guide, rng=rng)
            else:
                child = Individual.crossover_blend(parent_a, parent_b, config, guide, rng=rng)
            child.mutate(config, rate=mutation_rate * 0.75, guide=guide, rng=rng)
        else:
            child = parent_a.fork()
            child.mutate(config, rate=mutation_rate, guide=guide, rng=rng)
        offspring.append(child)

    return offspring[: config.pop_size]
