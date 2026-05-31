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

    def elite_fraction(self) -> float:
        return self.nb_elite / max(self.pop_size, 1)
