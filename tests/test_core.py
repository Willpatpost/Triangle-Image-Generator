from __future__ import annotations

import tempfile
import unittest
import random
from pathlib import Path

import numpy as np

from core.config import Config
from core.fitness import normalized_mse, shape_penalty
from core.io import load_state, save_state
from core.renderer import render_individual
from core.shapes import Circle, Individual, Square, Triangle, VoronoiSite, mutate_triangle


class ConfigTests(unittest.TestCase):
    def test_color_is_default_mode(self) -> None:
        self.assertEqual(Config().mode, "color")


class FitnessTests(unittest.TestCase):
    def test_normalized_mse_is_independent_of_image_size(self) -> None:
        small_ref = np.zeros((10, 10), dtype=np.uint8)
        small_rendered = np.full((10, 10), 255, dtype=np.uint8)
        large_ref = np.zeros((100, 100), dtype=np.uint8)
        large_rendered = np.full((100, 100), 255, dtype=np.uint8)

        self.assertAlmostEqual(normalized_mse(small_ref, small_rendered), 1.0)
        self.assertAlmostEqual(normalized_mse(large_ref, large_rendered), 1.0)

    def test_shape_penalty_increases_with_triangle_count(self) -> None:
        config = Config(shape_penalty_weight=0.01, nb_elements_max=100)

        self.assertLess(shape_penalty(10, config), shape_penalty(50, config))
        self.assertAlmostEqual(shape_penalty(100, config), 0.01)


class StateTests(unittest.TestCase):
    def test_save_load_state_roundtrip_preserves_metadata(self) -> None:
        individual = Individual(
            width=2,
            height=2,
            mode="color",
            triangles=[
                Triangle(
                    points=np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int32),
                    color=(10, 20, 30),
                    alpha=128,
                )
            ],
            fitness=0.25,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            save_state(
                str(path),
                iteration=7,
                best=individual,
                best_fitness=0.25,
                config=Config(mode="color"),
                background=(1, 2, 3),
                source_image="example.png",
                downsample=2,
            )

            iteration, fitness, loaded, config, background = load_state(str(path))

        self.assertEqual(iteration, 7)
        self.assertEqual(fitness, 0.25)
        self.assertEqual(loaded.width, 2)
        self.assertEqual(loaded.height, 2)
        self.assertEqual(loaded.mode, "color")
        self.assertEqual(loaded.triangles[0].color, (10, 20, 30))
        self.assertEqual(config.mode, "color")
        self.assertEqual(background, (1, 2, 3))

    def test_save_load_state_roundtrip_preserves_mixed_shapes(self) -> None:
        individual = Individual(
            width=8,
            height=8,
            mode="grayscale",
            triangles=[
                Circle(center=np.array([3, 3], dtype=np.int32), radius=2, color=80, alpha=200),
                Square(top_left=np.array([1, 1], dtype=np.int32), side=3, color=120, alpha=150),
                Triangle(
                    points=np.array([[0, 0], [7, 0], [0, 7]], dtype=np.int32),
                    color=200,
                    alpha=100,
                ),
                VoronoiSite(point=np.array([7, 7], dtype=np.int32), color=40),
            ],
            fitness=0.1,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            save_state(
                str(path),
                iteration=1,
                best=individual,
                best_fitness=0.1,
                config=Config(shape_mode="mixed"),
                background=255,
            )
            _, _, loaded, config, _ = load_state(str(path))

        self.assertIsInstance(loaded.triangles[0], Circle)
        self.assertIsInstance(loaded.triangles[1], Square)
        self.assertIsInstance(loaded.triangles[2], Triangle)
        self.assertIsInstance(loaded.triangles[3], VoronoiSite)
        self.assertEqual(config.shape_mode, "mixed")


class RendererTests(unittest.TestCase):
    def test_renders_circle_and_square(self) -> None:
        individual = Individual(
            width=8,
            height=8,
            mode="grayscale",
            triangles=[
                Circle(center=np.array([4, 4], dtype=np.int32), radius=2, color=0, alpha=255),
                Square(top_left=np.array([0, 0], dtype=np.int32), side=2, color=128, alpha=255),
            ],
        )

        rendered = render_individual(individual, Config(mode="grayscale", shape_mode="mixed"), 255)

        self.assertEqual(rendered.shape, (8, 8))
        self.assertEqual(rendered[4, 4], 0)
        self.assertEqual(rendered[0, 0], 128)

    def test_color_mutation_changes_at_most_one_rgb_channel_by_default(self) -> None:
        random.seed(1)
        triangle = Triangle(
            points=np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int32),
            color=(100, 100, 100),
            alpha=100,
        )
        config = Config(
            mode="color",
            mutation_sigma_points=0,
            mutation_sigma_color=25,
            mutation_sigma_alpha=25,
        )

        mutate_triangle(triangle, width=4, height=4, mode="color", config=config, rate=1.0)

        assert isinstance(triangle.color, tuple)
        changed_channels = sum(value != 100 for value in triangle.color)
        self.assertLessEqual(changed_channels, 1)

    def test_renders_voronoi_sites_by_nearest_point(self) -> None:
        individual = Individual(
            width=6,
            height=2,
            mode="grayscale",
            triangles=[
                VoronoiSite(point=np.array([0, 0], dtype=np.int32), color=10),
                VoronoiSite(point=np.array([5, 0], dtype=np.int32), color=240),
            ],
        )

        rendered = render_individual(individual, Config(mode="grayscale", shape_mode="voronoi"), 255)

        self.assertEqual(rendered.shape, (2, 6))
        self.assertEqual(rendered[0, 0], 10)
        self.assertEqual(rendered[0, 5], 240)


if __name__ == "__main__":
    unittest.main()
