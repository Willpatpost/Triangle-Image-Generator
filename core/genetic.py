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
) -> list[Individual]:
    population: list[Individual] = []
    if seed is not None:
        seeded = seed.copy()
        seeded.fitness = seed.fitness if seed._pixel_error_sum is not None else float("inf")
        population.append(seeded)
    while len(population) < config.pop_size:
        population.append(
            Individual.random(width, height, mode, config, config.nb_elements_initial, guide)
        )
    return population


def hill_climb_elite(
    elite: Individual,
    config: Config,
    rate: float = 0.5,
    guide: EvolutionGuide | None = None,
) -> Individual:
    candidate = elite.fork()
    candidate.mutate(config, rate=config.mutation_rate * rate, guide=guide)
    candidate.fitness = float("inf")
    return candidate


def _select_elite(
    population: list[Individual],
    ranked: list[tuple[int, float]],
    elite_count: int,
) -> Individual:
    # Squaring biases selection toward the best ranks while preserving diversity.
    rank = min(elite_count - 1, int(random.random() ** 2 * elite_count))
    return population[ranked[rank][0]]


def next_generation(
    population: list[Individual],
    ranked: list[tuple[int, float]],
    config: Config,
    mutation_rate: float,
    crossover_generation: bool,
    guide: EvolutionGuide | None = None,
) -> list[Individual]:
    elite_count = min(config.nb_elite, len(ranked))
    elite_sources = [population[ranked[index][0]] for index in range(elite_count)]
    offspring = [elite.fork() for elite in elite_sources]

    crossover_weight = config.prob_uniform_crossover + config.prob_blend_crossover
    while len(offspring) < config.pop_size:
        parent_a = _select_elite(population, ranked, elite_count)
        use_crossover = (
            crossover_generation
            and elite_count > 1
            and crossover_weight > 0
            and random.random() < config.crossover_rate
        )
        if use_crossover:
            parent_b = _select_elite(population, ranked, elite_count)
            uniform_probability = config.prob_uniform_crossover / crossover_weight
            if random.random() < uniform_probability:
                child = Individual.crossover_uniform(parent_a, parent_b, config, guide)
            else:
                child = Individual.crossover_blend(parent_a, parent_b, config, guide)
            child.mutate(config, rate=mutation_rate * 0.75, guide=guide)
        else:
            child = parent_a.fork()
            child.mutate(config, rate=mutation_rate, guide=guide)
        offspring.append(child)

    return offspring[: config.pop_size]
