from __future__ import annotations

import tempfile
import unittest
import random
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from core.acceleration import (
    AcceleratorUnavailable,
    _render_voronoi_color_impl,
    _render_color_impl,
    acceleration_status,
    available_renderer_backends,
    encode_shapes,
    resolve_renderer_backend,
)
from core.config import Config
from core.evolver import EvolutionSession
from core.fitness import evaluate_fitness, evaluate_population, normalized_mse, shape_penalty
from core.genetic import next_generation
from core.guidance import EvolutionGuide
from core.io import load_state, read_state, save_state
from core.pipeline import run_evolution, run_progressive_evolution
from core.renderer import (
    new_canvas,
    render_individual,
    render_individual_float,
    render_individual_for_scoring,
)
from core.shapes import (
    Circle,
    Individual,
    Square,
    Triangle,
    VoronoiSite,
    mutate_triangle,
    random_circle,
    random_square,
    random_triangle,
    scale_individual,
)


class ConfigTests(unittest.TestCase):
    def test_color_is_default_mode(self) -> None:
        self.assertEqual(Config().mode, "color")

    def test_quality_defaults_do_not_blur_or_penalize_the_target(self) -> None:
        config = Config()

        self.assertEqual(config.gaussian_blur_sigma, 0.0)
        self.assertEqual(config.shape_penalty_weight, 0.0)

    def test_validate_rejects_invalid_population_settings(self) -> None:
        with self.assertRaises(ValueError):
            Config(pop_size=1).validate()
        with self.assertRaises(ValueError):
            Config(pop_size=4, nb_elite=4).validate()
        with self.assertRaises(ValueError):
            Config(nb_elements_initial=10, nb_elements_max=9).validate()

    def test_validate_rejects_invalid_alpha_settings(self) -> None:
        with self.assertRaises(ValueError):
            Config(alpha_min=-1).validate()
        with self.assertRaises(ValueError):
            Config(alpha_min=200, alpha_max=100).validate()
        with self.assertRaises(ValueError):
            Config(fixed_alpha=1.1).validate()
        with self.assertRaises(ValueError):
            Config(new_shape_size_divisor=0.5).validate()
        with self.assertRaises(ValueError):
            Config(compositing_cache_stride=0).validate()
        with self.assertRaises(ValueError):
            Config(prob_structural=0.9, prob_reorder=0.2).validate()
        with self.assertRaises(ValueError):
            Config(renderer_backend="metal").validate()  # type: ignore[arg-type]


class FitnessTests(unittest.TestCase):
    def test_normalized_mse_is_independent_of_image_size(self) -> None:
        small_ref = np.zeros((10, 10), dtype=np.uint8)
        small_rendered = np.full((10, 10), 255, dtype=np.uint8)
        large_ref = np.zeros((100, 100), dtype=np.uint8)
        large_rendered = np.full((100, 100), 255, dtype=np.uint8)

        self.assertAlmostEqual(normalized_mse(small_ref, small_rendered), 1.0)
        self.assertAlmostEqual(normalized_mse(large_ref, large_rendered), 1.0)

    def test_shape_penalty_increases_with_shape_count(self) -> None:
        config = Config(shape_penalty_weight=0.01, nb_elements_max=100)

        self.assertLess(shape_penalty(10, config), shape_penalty(50, config))
        self.assertAlmostEqual(shape_penalty(100, config), 0.01)

    def test_population_retains_render_caches_only_for_elites(self) -> None:
        population = [
            Individual(
                width=4,
                height=4,
                mode="grayscale",
                shapes=[
                    Triangle(
                        points=np.array([[0, 0], [3, 0], [0, 3]], dtype=np.int32),
                        color=color,
                        alpha=255,
                    )
                ],
            )
            for color in (0, 60, 120, 240)
        ]
        config = Config(
            mode="grayscale",
            pop_size=4,
            nb_elite=1,
            nb_elements_initial=1,
            nb_elements_max=2,
            min_shapes=1,
        )

        ranked = evaluate_population(population, np.zeros((4, 4), dtype=np.uint8), config, 0)

        best_index = ranked[0][0]
        self.assertTrue(population[best_index]._compositing_cache)
        for index, individual in enumerate(population):
            if index != best_index:
                self.assertEqual(individual._compositing_cache, [])
                self.assertIsNone(individual.shapes[0]._mask_cache)

    def test_dirty_region_render_and_fitness_match_full_recompute(self) -> None:
        reference = np.random.default_rng(7).integers(0, 256, (20, 24, 3), dtype=np.uint8)
        parent = Individual(
            width=24,
            height=20,
            mode="color",
            shapes=[
                Triangle(
                    points=np.array([[1, 1], [20, 2], [4, 17]], dtype=np.int32),
                    color=(30, 80, 210),
                    alpha=130,
                ),
                Circle(
                    center=np.array([10, 9], dtype=np.int32),
                    radius=5,
                    color=(220, 40, 90),
                    alpha=180,
                ),
                Square(
                    top_left=np.array([7, 4], dtype=np.int32),
                    side=8,
                    color=(20, 230, 70),
                    alpha=210,
                ),
            ],
        )
        config = Config(
            mode="color",
            shape_mode="mixed",
            pop_size=4,
            nb_elite=1,
            nb_elements_initial=3,
            nb_elements_max=8,
            min_shapes=1,
            renderer_backend="numpy",
            compositing_cache_stride=1,
        )
        background = (110, 120, 130)
        evaluate_fitness(parent, reference, config, background)

        child = parent.fork()
        circle = child.shapes[1]
        assert isinstance(circle, Circle)
        child._mark_dirty(1, circle.bounding_box(child.width, child.height))
        circle.center = circle.center.copy()
        circle.center[0] += 3
        circle.color = (190, 70, 120)
        circle.invalidate_caches()
        child._mark_dirty(1, circle.bounding_box(child.width, child.height))
        child.invalidate_cache_from(1)
        child.fitness = float("inf")

        probe = child.copy()
        render_result = render_individual_for_scoring(probe, config, background)
        self.assertTrue(render_result.incremental)
        self.assertLess(
            (render_result.dirty_bbox[2] - render_result.dirty_bbox[0] + 1)
            * (render_result.dirty_bbox[3] - render_result.dirty_bbox[1] + 1),
            child.width * child.height,
        )

        full = child.copy()
        full.clear_caches()
        incremental_fitness = evaluate_fitness(child, reference, config, background)
        full_config = replace(config, use_dirty_regions=False, use_compositing_cache=False)
        full_fitness = evaluate_fitness(full, reference, full_config, background)

        np.testing.assert_allclose(
            render_individual_float(child, config, background),
            render_individual_float(full, full_config, background),
            atol=1e-6,
        )
        self.assertAlmostEqual(incremental_fitness, full_fitness, places=12)

        removed = parent.fork()
        with patch("core.shapes.random.randint", return_value=1):
            self.assertTrue(removed.remove_random_shape(config))
        removed_full = removed.copy()
        removed_full.clear_caches()
        removed_result = render_individual_for_scoring(removed.copy(), config, background)
        self.assertTrue(removed_result.incremental)
        removed_fitness = evaluate_fitness(removed, reference, config, background)
        removed_full_fitness = evaluate_fitness(
            removed_full,
            reference,
            full_config,
            background,
        )
        np.testing.assert_allclose(
            render_individual_float(removed, config, background),
            render_individual_float(removed_full, full_config, background),
            atol=1e-6,
        )
        self.assertAlmostEqual(removed_fitness, removed_full_fitness, places=12)


class AccelerationTests(unittest.TestCase):
    @staticmethod
    def _sample_individual() -> Individual:
        return Individual(
            width=24,
            height=18,
            mode="color",
            shapes=[
                Triangle(
                    points=np.array([[1, 2], [20, 4], [6, 16]], dtype=np.int32),
                    color=(20, 100, 220),
                    alpha=101,
                ),
                Circle(
                    center=np.array([15, 10], dtype=np.int32),
                    radius=6,
                    color=(220, 50, 80),
                    alpha=177,
                ),
                Square(
                    top_left=np.array([5, 7], dtype=np.int32),
                    side=8,
                    color=(30, 230, 40),
                    alpha=219,
                ),
            ],
        )

    def test_acceleration_capabilities_always_include_numpy(self) -> None:
        status = acceleration_status()
        self.assertIn("numpy", available_renderer_backends())
        self.assertEqual(
            resolve_renderer_backend(
                "auto",
                1,
                Config(numba_min_pixels=2, cuda_min_pixels=2),
            ),
            "numpy",
        )
        if not status.numba_available:
            with self.assertRaises(AcceleratorUnavailable):
                resolve_renderer_backend("numba", 10_000, Config())
        if not status.cuda_available:
            with self.assertRaises(AcceleratorUnavailable):
                resolve_renderer_backend("cuda", 10_000, Config())

    def test_auto_backend_runtime_failure_falls_back_to_numpy(self) -> None:
        individual = Individual(
            width=8,
            height=8,
            mode="grayscale",
            shapes=[
                Square(top_left=np.array([1, 1], dtype=np.int32), side=4, color=30, alpha=190)
            ],
        )
        config = Config(
            mode="grayscale",
            shape_mode="square",
            renderer_backend="auto",
            use_compositing_cache=False,
        )
        expected = render_individual_float(
            individual.copy(),
            replace(config, renderer_backend="numpy"),
            220,
        )

        with (
            patch("core.renderer.resolve_renderer_backend", return_value="numba"),
            patch("core.renderer.render_shapes_accelerated", side_effect=RuntimeError("jit failed")),
            patch("core.renderer.mark_renderer_backend_failed", return_value=True),
            self.assertLogs(level="WARNING"),
        ):
            actual = render_individual_float(individual, config, 220)

        np.testing.assert_array_equal(actual, expected)

    def test_numba_loop_body_matches_numpy_renderer(self) -> None:
        individual = self._sample_individual()
        config = Config(
            mode="color",
            shape_mode="mixed",
            renderer_backend="numpy",
            use_compositing_cache=False,
        )
        background = (90, 120, 150)
        expected = render_individual_float(individual, config, background)
        encoded = encode_shapes(individual, config)
        actual = new_canvas(individual.width, individual.height, individual.mode, background)

        _render_color_impl(
            actual,
            encoded.kinds,
            encoded.geometry,
            encoded.bounds,
            encoded.colors,
            encoded.alphas,
            0,
            0,
            0,
        )

        np.testing.assert_allclose(actual, expected, atol=2e-5)

    def test_numba_renderer_matches_numpy_when_available(self) -> None:
        if not acceleration_status().numba_available:
            self.skipTest("Numba is not installed")
        individual = self._sample_individual()
        base_config = Config(
            mode="color",
            shape_mode="mixed",
            use_compositing_cache=False,
        )
        expected = render_individual_float(
            individual.copy(),
            replace(base_config, renderer_backend="numpy"),
            (90, 120, 150),
        )
        actual = render_individual_float(
            individual,
            replace(base_config, renderer_backend="numba"),
            (90, 120, 150),
        )
        np.testing.assert_allclose(actual, expected, atol=2e-5)

    def test_cuda_renderer_matches_numpy_when_available(self) -> None:
        if not acceleration_status().cuda_available:
            self.skipTest("CUDA is not available")
        individual = self._sample_individual()
        base_config = Config(
            mode="color",
            shape_mode="mixed",
            use_compositing_cache=False,
        )
        expected = render_individual_float(
            individual.copy(),
            replace(base_config, renderer_backend="numpy"),
            (90, 120, 150),
        )
        actual = render_individual_float(
            individual,
            replace(base_config, renderer_backend="cuda"),
            (90, 120, 150),
        )
        np.testing.assert_allclose(actual, expected, atol=2e-4)

    def test_numba_voronoi_loop_body_matches_numpy_renderer(self) -> None:
        individual = Individual(
            width=12,
            height=8,
            mode="color",
            shapes=[
                VoronoiSite(point=np.array([0, 0], dtype=np.int32), color=(10, 20, 30)),
                VoronoiSite(point=np.array([11, 7], dtype=np.int32), color=(230, 210, 190)),
                VoronoiSite(point=np.array([6, 2], dtype=np.int32), color=(80, 180, 60)),
            ],
        )
        config = Config(
            mode="color",
            shape_mode="voronoi",
            renderer_backend="numpy",
            use_compositing_cache=False,
        )
        background = (100, 100, 100)
        expected = render_individual_float(individual, config, background)
        encoded = encode_shapes(individual, config)
        actual = new_canvas(individual.width, individual.height, individual.mode, background)

        _render_voronoi_color_impl(actual, encoded.geometry, encoded.colors)

        np.testing.assert_array_equal(actual, expected)


class StateTests(unittest.TestCase):
    def test_save_load_state_roundtrip_preserves_metadata(self) -> None:
        individual = Individual(
            width=2,
            height=2,
            mode="color",
            shapes=[
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
            state = read_state(str(path))
            temporary_files = list(Path(tmp).glob(".*.tmp"))

        self.assertEqual(iteration, 7)
        self.assertEqual(fitness, 0.25)
        self.assertEqual(loaded.width, 2)
        self.assertEqual(loaded.height, 2)
        self.assertEqual(loaded.mode, "color")
        self.assertEqual(loaded.shapes[0].color, (10, 20, 30))
        self.assertEqual(config.mode, "color")
        self.assertEqual(background, (1, 2, 3))
        self.assertEqual(state.version, 2)
        self.assertEqual(state.downsample, 2)
        self.assertEqual(temporary_files, [])

    def test_save_load_state_roundtrip_preserves_mixed_shapes(self) -> None:
        individual = Individual(
            width=8,
            height=8,
            mode="grayscale",
            shapes=[
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

        self.assertIsInstance(loaded.shapes[0], Circle)
        self.assertIsInstance(loaded.shapes[1], Square)
        self.assertIsInstance(loaded.shapes[2], Triangle)
        self.assertIsInstance(loaded.shapes[3], VoronoiSite)
        self.assertEqual(config.shape_mode, "mixed")

    def test_loads_legacy_triangle_keyed_state(self) -> None:
        payload = {
            "iteration": 4,
            "best_fitness": 0.2,
            "width": 2,
            "height": 2,
            "mode": "grayscale",
            "background": 255,
            "config": {"mode": "grayscale", "min_triangles": 2},
            "triangles": [
                {
                    "kind": "square",
                    "top_left": [0, 0],
                    "side": 1,
                    "color": 10,
                    "alpha": 255,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            state = read_state(str(path))

        self.assertEqual(state.version, 1)
        self.assertEqual(state.config.min_shapes, 2)
        self.assertIsInstance(state.best.shapes[0], Square)

    def test_rejects_state_from_a_newer_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "future.json"
            path.write_text('{"version": 999}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "newer than"):
                read_state(str(path))


class RendererTests(unittest.TestCase):
    def test_guidance_solves_new_shape_color_against_current_canvas(self) -> None:
        reference = np.array([[[100, 120, 140]]], dtype=np.uint8)
        baseline = np.array([[[0, 20, 40]]], dtype=np.float32)
        guide = EvolutionGuide(reference, baseline)

        self.assertEqual(guide.sample_high_error_point(4), (0, 0))
        self.assertEqual(guide.optimal_color(0, 0, 0.5, "color"), (200, 220, 240))

    def test_random_shapes_use_size_divisor(self) -> None:
        random.seed(2)
        config = Config(mode="grayscale", new_shape_size_divisor=5.0)
        triangle = random_triangle(100, 100, "grayscale", config)
        circle = random_circle(100, 100, "grayscale", config)
        square = random_square(100, 100, "grayscale", config)

        min_x, min_y, max_x, max_y = triangle.bounding_box(100, 100)
        self.assertLessEqual(max_x - min_x, 40)
        self.assertLessEqual(max_y - min_y, 40)
        self.assertLessEqual(circle.radius, 20)
        self.assertLessEqual(square.side, 20)

    def test_renders_circle_and_square(self) -> None:
        individual = Individual(
            width=8,
            height=8,
            mode="grayscale",
            shapes=[
                Circle(center=np.array([4, 4], dtype=np.int32), radius=2, color=0, alpha=255),
                Square(top_left=np.array([0, 0], dtype=np.int32), side=2, color=128, alpha=255),
            ],
        )

        rendered = render_individual(individual, Config(mode="grayscale", shape_mode="mixed"), 255)

        self.assertEqual(rendered.shape, (8, 8))
        self.assertEqual(rendered[4, 4], 0)
        self.assertEqual(rendered[0, 0], 128)

    def test_final_render_rounds_float_canvas_to_nearest_byte(self) -> None:
        individual = Individual(
            width=1,
            height=1,
            mode="grayscale",
            shapes=[
                Square(top_left=np.array([0, 0], dtype=np.int32), side=1, color=100, alpha=255)
            ],
        )

        rendered = render_individual(
            individual,
            Config(mode="grayscale", shape_mode="square", fixed_alpha=0.5),
            103,
        )

        self.assertEqual(rendered[0, 0], 102)

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
            shapes=[
                VoronoiSite(point=np.array([0, 0], dtype=np.int32), color=10),
                VoronoiSite(point=np.array([5, 0], dtype=np.int32), color=240),
            ],
        )

        rendered = render_individual(individual, Config(mode="grayscale", shape_mode="voronoi"), 255)

        self.assertEqual(rendered.shape, (2, 6))
        self.assertEqual(rendered[0, 0], 10)
        self.assertEqual(rendered[0, 5], 240)

    def test_compositing_cache_survives_copy_and_partial_invalidation(self) -> None:
        individual = Individual(
            width=6,
            height=6,
            mode="grayscale",
            shapes=[
                Square(top_left=np.array([0, 0], dtype=np.int32), side=5, color=200, alpha=255),
                Square(top_left=np.array([2, 2], dtype=np.int32), side=2, color=50, alpha=255),
            ],
        )
        cached_config = Config(
            mode="grayscale",
            shape_mode="square",
            use_compositing_cache=True,
            compositing_cache_stride=1,
        )

        render_individual(individual, cached_config, 255)
        copied = individual.copy()
        assert isinstance(copied.shapes[1], Square)
        copied.shapes[1].color = 10
        copied.invalidate_cache_from(1)

        cached = render_individual(copied, cached_config, 255)
        self.assertIsNotNone(copied._compositing_cache[0])
        self.assertIsNotNone(copied._compositing_cache[1])

        copied._compositing_cache = []
        uncached = render_individual(copied, Config(mode="grayscale", shape_mode="square"), 255)

        np.testing.assert_array_equal(cached, uncached)

    def test_compositing_cache_uses_sparse_checkpoints(self) -> None:
        individual = Individual(
            width=8,
            height=8,
            mode="grayscale",
            shapes=[
                Square(top_left=np.array([0, 0], dtype=np.int32), side=1, color=i, alpha=128)
                for i in range(10)
            ],
        )
        config = Config(
            mode="grayscale",
            shape_mode="square",
            use_compositing_cache=True,
            compositing_cache_stride=4,
        )

        render_individual(individual, config, 255)

        cached_indices = [i for i, canvas in enumerate(individual._compositing_cache) if canvas is not None]
        self.assertEqual(cached_indices, [3, 7, 9])

    def test_compositing_cache_obeys_per_individual_memory_budget(self) -> None:
        individual = Individual(
            width=16,
            height=16,
            mode="grayscale",
            shapes=[
                Square(top_left=np.array([0, 0], dtype=np.int32), side=1, color=i, alpha=128)
                for i in range(10)
            ],
        )
        config = Config(
            mode="grayscale",
            shape_mode="square",
            compositing_cache_stride=1,
            compositing_cache_max_mb=0.0001,
        )

        render_individual(individual, config, 255)

        cached_indices = [i for i, canvas in enumerate(individual._compositing_cache) if canvas is not None]
        self.assertEqual(cached_indices, [9])

    def test_shape_mask_is_shared_until_geometry_changes(self) -> None:
        triangle = Triangle(
            points=np.array([[0, 0], [5, 0], [0, 5]], dtype=np.int32),
            color=0,
            alpha=255,
        )
        individual = Individual(width=6, height=6, mode="grayscale", shapes=[triangle])
        config = Config(mode="grayscale", shape_mode="triangle")
        render_individual(individual, config, 255)
        copied = individual.copy()

        self.assertIs(copied.shapes[0]._mask_cache, triangle._mask_cache)
        mutate_triangle(
            copied.shapes[0],
            6,
            6,
            "grayscale",
            Config(mode="grayscale", prob_mutate_geometry=1.0),
            1.0,
        )
        self.assertIsNone(copied.shapes[0]._mask_cache)

    def test_shape_genome_scales_to_finer_resolution(self) -> None:
        individual = Individual(
            width=4,
            height=4,
            mode="grayscale",
            shapes=[
                Triangle(
                    points=np.array([[0, 0], [3, 0], [0, 3]], dtype=np.int32),
                    color=20,
                    alpha=100,
                ),
                Circle(center=np.array([2, 2], dtype=np.int32), radius=1, color=40, alpha=120),
                Square(top_left=np.array([1, 1], dtype=np.int32), side=2, color=60, alpha=140),
                VoronoiSite(point=np.array([3, 3], dtype=np.int32), color=80),
            ],
            fitness=0.1,
        )

        scaled = scale_individual(individual, 8, 8)

        self.assertEqual((scaled.width, scaled.height), (8, 8))
        self.assertEqual(scaled.fitness, float("inf"))
        np.testing.assert_array_equal(
            scaled.shapes[0].points,
            np.array([[0, 0], [7, 0], [0, 7]], dtype=np.int32),
        )
        np.testing.assert_array_equal(scaled.shapes[3].point, np.array([7, 7], dtype=np.int32))


class GeneticTests(unittest.TestCase):
    def test_internal_fork_copies_geometry_only_when_mutated(self) -> None:
        parent = Individual(
            width=8,
            height=8,
            mode="grayscale",
            shapes=[
                Triangle(
                    points=np.array([[1, 1], [6, 1], [1, 6]], dtype=np.int32),
                    color=80,
                    alpha=180,
                )
            ],
        )
        render_individual(parent, Config(mode="grayscale"), 255)
        original_points = parent.shapes[0].points.copy()
        child = parent.fork()

        self.assertIs(child.shapes[0].points, parent.shapes[0].points)
        self.assertIs(child.shapes[0]._mask_cache, parent.shapes[0]._mask_cache)

        child.mutate(
            Config(
                mode="grayscale",
                nb_elements_initial=1,
                nb_elements_max=1,
                min_shapes=1,
                prob_structural=0.0,
                prob_reorder=0.0,
                prob_mutate_geometry=1.0,
            ),
            rate=1.0,
        )

        self.assertIsNot(child.shapes[0].points, parent.shapes[0].points)
        np.testing.assert_array_equal(parent.shapes[0].points, original_points)
        self.assertIsNone(child.shapes[0]._mask_cache)
        self.assertIsNotNone(parent.shapes[0]._mask_cache)

    def test_next_generation_reuses_unchanged_elite_fitness(self) -> None:
        population = [
            Individual(
                width=2,
                height=2,
                mode="grayscale",
                shapes=[
                    Square(
                        top_left=np.array([0, 0], dtype=np.int32),
                        side=1,
                        color=50 + index,
                        alpha=255,
                    )
                ],
                fitness=0.1 + index,
            )
            for index in range(4)
        ]
        ranked = [(index, individual.fitness) for index, individual in enumerate(population)]
        config = Config(
            mode="grayscale",
            shape_mode="square",
            pop_size=4,
            nb_elite=1,
            nb_elements_initial=1,
            nb_elements_max=2,
            min_shapes=1,
            prob_structural=0.0,
            prob_reorder=0.0,
            prob_mutate_geometry=0.0,
            crossover_rate=0.0,
        )

        offspring = next_generation(population, ranked, config, 1.0, True)

        self.assertEqual(offspring[0].fitness, 0.1)
        self.assertTrue(all(individual.fitness == float("inf") for individual in offspring[1:]))


class PipelineTests(unittest.TestCase):
    def test_evolution_session_steps_and_renders_without_saving(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "target.png"
            Image.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L").save(image_path)
            config = Config(
                mode="grayscale",
                shape_mode="triangle",
                pop_size=4,
                nb_elite=1,
                nb_elements_initial=2,
                nb_elements_max=8,
                min_shapes=1,
                hill_climb_interval=0,
                max_workers=1,
            )

            session = EvolutionSession(str(image_path), config)
            snapshot = session.step()
            rendered = session.render_best()

            self.assertEqual(snapshot.iteration, 1)
            self.assertEqual(rendered.shape, (4, 4))
            self.assertFalse((Path(tmp) / "final.png").exists())

    def test_returns_global_best_fitness_when_generation_regresses(self) -> None:
        initial_best = Individual(width=1, height=1, mode="grayscale", fitness=0.1)
        regressed = Individual(width=1, height=1, mode="grayscale", fitness=float("inf"))
        evaluated_regressed_generation = False

        def fake_evaluate_population(population, reference, config, background, max_workers):
            nonlocal evaluated_regressed_generation
            if not evaluated_regressed_generation and population[0].fitness == float("inf"):
                population[0].fitness = 0.2
                evaluated_regressed_generation = True
            return [(0, population[0].fitness)]

        with tempfile.TemporaryDirectory() as tmp:
            config = Config(
                mode="grayscale",
                pop_size=2,
                nb_elite=1,
                nb_elements_initial=1,
                nb_elements_max=1,
                min_shapes=1,
                max_workers=1,
                fitness_goal=0.0,
                stagnation_threshold=10,
            )

            with (
                patch(
                    "core.pipeline.load_reference_image",
                    return_value=(np.zeros((1, 1), dtype=np.uint8), 255),
                ),
                patch("core.pipeline.create_initial_population", return_value=[initial_best]),
                patch("core.pipeline.next_generation", return_value=[regressed]),
                patch("core.pipeline.evaluate_population_parallel", side_effect=fake_evaluate_population),
                patch("core.pipeline.render_individual", return_value=np.zeros((1, 1), dtype=np.uint8)),
            ):
                _, fitness = run_evolution("target.png", config, iterations=1)

        self.assertEqual(fitness, 0.1)

    def test_progressive_evolution_returns_final_resolution_without_saving(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "target.png"
            target = np.zeros((8, 8), dtype=np.uint8)
            target[:, 4:] = 255
            Image.fromarray(target, mode="L").save(image_path)
            config = Config(
                mode="grayscale",
                pop_size=4,
                nb_elite=1,
                nb_elements_initial=2,
                nb_elements_max=6,
                min_shapes=1,
                hill_climb_interval=0,
                max_workers=1,
            )

            rendered, fitness = run_progressive_evolution(
                str(image_path),
                config,
                stages=((2, 1), (1, 1)),
                seed=7,
            )

            self.assertEqual(rendered.shape, (8, 8))
            self.assertTrue(np.isfinite(fitness))
            self.assertFalse((Path(tmp) / "final.png").exists())

    def test_saved_session_resumes_the_same_evolution_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "target.png"
            target = np.zeros((8, 8), dtype=np.uint8)
            target[2:6, 2:6] = 220
            Image.fromarray(target, mode="L").save(image_path)
            config = Config(
                mode="grayscale",
                shape_mode="mixed",
                pop_size=4,
                nb_elite=1,
                nb_elements_initial=2,
                nb_elements_max=8,
                min_shapes=1,
                hill_climb_interval=0,
                max_workers=1,
            )
            session = EvolutionSession(str(image_path), config, seed=19)
            session.step_many(3)
            state_path = Path(tmp) / "session.json"
            session.save(str(state_path))

            expected = session.step()
            expected_image = session.render_best()
            resumed = EvolutionSession.from_state(str(state_path))
            actual = resumed.step()

            self.assertEqual(actual.iteration, expected.iteration)
            self.assertEqual(actual.shape_count, expected.shape_count)
            self.assertAlmostEqual(actual.current_fitness, expected.current_fitness, places=14)
            self.assertAlmostEqual(actual.best_fitness, expected.best_fitness, places=14)
            np.testing.assert_array_equal(resumed.render_best(), expected_image)

    def test_session_rejects_a_different_source_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.png"
            other = Path(tmp) / "other.png"
            Image.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L").save(source)
            Image.fromarray(np.full((4, 4), 255, dtype=np.uint8), mode="L").save(other)
            config = Config(
                mode="grayscale",
                pop_size=4,
                nb_elite=1,
                nb_elements_initial=1,
                nb_elements_max=2,
                min_shapes=1,
                hill_climb_interval=0,
                max_workers=1,
            )
            session = EvolutionSession(str(source), config, seed=3)
            state_path = Path(tmp) / "session.json"
            session.save(str(state_path))

            with self.assertRaisesRegex(ValueError, "does not match"):
                EvolutionSession.from_state(str(state_path), image_path=str(other))


if __name__ == "__main__":
    unittest.main()
