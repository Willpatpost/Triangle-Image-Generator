from __future__ import annotations

import random
from dataclasses import dataclass, replace

import numpy as np

from core.config import Config
from core.fitness import evaluate_population
from core.genetic import create_initial_population, hill_climb_elite, next_generation
from core.io import load_reference_image
from core.renderer import render_individual
from core.shapes import Individual


@dataclass(frozen=True)
class EvolutionSnapshot:
    iteration: int
    best_fitness: float
    current_fitness: float
    shape_count: int
    improved: bool


class EvolutionSession:
    def __init__(
        self,
        image_path: str,
        config: Config,
        *,
        downsample: int = 1,
        seed: int | None = None,
    ) -> None:
        if downsample < 1:
            raise ValueError("downsample must be at least 1")
        config.validate()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        reference, background = load_reference_image(
            image_path,
            config.mode,
            downsample,
            config.gaussian_blur_sigma,
        )
        self.reference = reference
        self.background = background
        self.config = replace(config, background=background, max_workers=1)
        self.iteration = 0
        self.stagnation = 0
        self.mutation_rate = self.config.mutation_rate

        height, width = reference.shape[:2]
        self.population = create_initial_population(width, height, self.config.mode, self.config)
        self.ranked = evaluate_population(self.population, self.reference, self.config, self.background)
        best_index, best_fitness = self.ranked[0]
        self.global_best = self.population[best_index].copy()
        self.global_best.fitness = best_fitness

    def step(self) -> EvolutionSnapshot:
        if self.stagnation > self.config.stagnation_threshold // 4:
            self.mutation_rate = min(self.config.mutation_rate * self.config.adaptive_mutation_boost, 3.0)
        else:
            self.mutation_rate = self.config.mutation_rate

        self.population = next_generation(
            self.population,
            self.ranked,
            self.config,
            self.mutation_rate,
            crossover_generation=self.iteration % 2 == 0,
        )

        if self.config.hill_climb_interval > 0 and self.iteration % self.config.hill_climb_interval == 0:
            seed_elite = self.population[0].copy()
            for _ in range(self.config.hill_climb_attempts):
                self.population.append(hill_climb_elite(seed_elite, self.config))

        for individual in self.population:
            individual.fitness = float("inf")

        self.ranked = evaluate_population(self.population, self.reference, self.config, self.background)
        if len(self.population) > self.config.pop_size:
            self.population = [self.population[i] for i, _ in self.ranked[: self.config.pop_size]]
            self.ranked = sorted(
                ((i, individual.fitness) for i, individual in enumerate(self.population)),
                key=lambda item: item[1],
            )

        best_index, current_fitness = self.ranked[0]
        improved = current_fitness < self.global_best.fitness
        if improved:
            self.global_best = self.population[best_index].copy()
            self.global_best.fitness = current_fitness
            self.stagnation = 0
        else:
            self.stagnation += 1

        self.iteration += 1
        return EvolutionSnapshot(
            iteration=self.iteration,
            best_fitness=self.global_best.fitness,
            current_fitness=current_fitness,
            shape_count=len(self.global_best.triangles),
            improved=improved,
        )

    def step_many(self, count: int) -> EvolutionSnapshot:
        if count < 1:
            raise ValueError("count must be at least 1")
        snapshot = self.step()
        for _ in range(count - 1):
            snapshot = self.step()
        return snapshot

    def render_best(self) -> np.ndarray:
        return render_individual(self.global_best, self.config, self.background)

    def best_individual(self) -> Individual:
        return self.global_best.copy()
