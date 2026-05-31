from __future__ import annotations

import copy
import random

from core.config import ColorMode, Config
from core.shapes import Individual


def create_initial_population(
    width: int,
    height: int,
    mode: ColorMode,
    config: Config,
    seed: Individual | None = None,
) -> list[Individual]:
    population = [
        Individual.random(width, height, mode, config, config.nb_elements_initial)
        for _ in range(config.pop_size)
    ]
    if seed is not None:
        population[0] = seed.copy()
        population[0].fitness = seed.fitness
    return population


def hill_climb_elite(elite: Individual, config: Config, rate: float = 0.5) -> Individual:
    candidate = elite.copy()
    candidate.mutate(config, rate=rate)
    candidate.fitness = float("inf")
    return candidate


def next_generation(
    population: list[Individual],
    ranked: list[tuple[int, float]],
    config: Config,
    mutation_rate: float,
    crossover_generation: bool,
) -> list[Individual]:
    elites = [copy.deepcopy(population[ranked[i][0]]) for i in range(config.nb_elite)]
    offspring: list[Individual] = []

    for elite in elites:
        offspring.append(elite.copy())

    if crossover_generation:
        while len(offspring) < config.pop_size:
            parent_a = population[ranked[random.randint(0, config.nb_elite - 1)][0]]
            parent_b = population[ranked[random.randint(0, config.nb_elite - 1)][0]]
            if random.random() < config.prob_blend_crossover:
                child = Individual.crossover_blend(parent_a, parent_b, config)
            else:
                child = Individual.crossover_uniform(parent_a, parent_b, config)
            child.mutate(config, rate=mutation_rate * 0.5)
            offspring.append(child)
    else:
        for elite in elites:
            mutant = elite.copy()
            mutant.mutate(config, rate=mutation_rate)
            offspring.append(mutant)

        while len(offspring) < config.pop_size:
            parent = population[ranked[random.randint(0, config.nb_elite - 1)][0]]
            child = parent.copy()
            child.mutate(config, rate=mutation_rate)
            offspring.append(child)

    return offspring[: config.pop_size]
