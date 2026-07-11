from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from core.paths import OUTPUT_DIR


ColorMode = Literal["grayscale", "color"]
ShapeMode = Literal["triangle", "circle", "square", "voronoi", "mixed"]


@dataclass
class Config:
    mode: ColorMode = "color"
    shape_mode: ShapeMode = "triangle"

    # Population
    pop_size: int = 30
    nb_elite: int = 6
    nb_elements_initial: int = 15
    nb_elements_max: int = 150
    min_triangles: int = 5

    # Mutation
    mutation_rate: float = 1.0
    prob_structural: float = 0.25
    prob_add_vs_del: float = 0.92
    prob_reorder: float = 0.08
    mutation_sigma_points: float = 10.0
    mutation_sigma_color: float = 12.0
    mutation_sigma_alpha: float = 10.0
    mutate_all_color_channels: bool = False

    # Crossover
    prob_uniform_crossover: float = 0.5
    prob_blend_crossover: float = 0.5

    # Rendering
    fixed_alpha: float = -1.0  # <0 uses per-triangle alpha
    alpha_min: int = 10
    alpha_max: int = 245
    gaussian_blur_sigma: float = 0.5
    use_compositing_cache: bool = False
    new_shape_size_divisor: float = 4.0

    # Fitness
    shape_penalty_weight: float = 0.01
    fitness_goal: float = 0.001

    # Evolution
    stagnation_threshold: int = 5000
    adaptive_mutation_boost: float = 1.5
    hill_climb_interval: int = 100
    hill_climb_attempts: int = 3

    # Parallelism
    max_workers: int = 0  # 0 = auto (cpu_count - 1, min 1)

    # I/O
    save_directory: str = str(OUTPUT_DIR)
    save_comparison: bool = True
    gif_frame_duration_ms: int = 200

    # Logging
    enable_logging: bool = True
    log_interval_sec: float = 10.0
    log_level: str = "INFO"

    # Runtime (set during execution)
    background: int | tuple[int, int, int] = field(default=255, repr=False)

    def validate(self) -> None:
        if self.pop_size < 2:
            raise ValueError("pop_size must be at least 2")
        if self.nb_elite < 1:
            raise ValueError("nb_elite must be at least 1")
        if self.nb_elite >= self.pop_size:
            raise ValueError("nb_elite must be smaller than pop_size")
        if self.nb_elements_initial < 1:
            raise ValueError("nb_elements_initial must be at least 1")
        if self.nb_elements_max < self.nb_elements_initial:
            raise ValueError("nb_elements_max must be greater than or equal to nb_elements_initial")
        if self.min_triangles < 1:
            raise ValueError("min_triangles must be at least 1")
        if self.nb_elements_max < self.min_triangles:
            raise ValueError("nb_elements_max must be greater than or equal to min_triangles")
        if self.max_workers < 0:
            raise ValueError("max_workers must be 0 or greater")
        if not 0 <= self.alpha_min <= 255:
            raise ValueError("alpha_min must be between 0 and 255")
        if not 0 <= self.alpha_max <= 255:
            raise ValueError("alpha_max must be between 0 and 255")
        if self.alpha_min > self.alpha_max:
            raise ValueError("alpha_min must be less than or equal to alpha_max")
        if self.fixed_alpha > 1.0:
            raise ValueError("fixed_alpha must be negative or between 0.0 and 1.0")
        if self.new_shape_size_divisor < 1.0:
            raise ValueError("new_shape_size_divisor must be at least 1.0")

    def elite_fraction(self) -> float:
        return self.nb_elite / max(self.pop_size, 1)
