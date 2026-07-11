from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from core.acceleration import (
    mark_renderer_backend_failed,
    render_shapes_accelerated,
    render_voronoi_accelerated,
    resolve_renderer_backend,
)
from core.config import ColorMode, Config
from core.shapes import BBox, Circle, Individual, Shape, Square, Triangle, VoronoiSite


@dataclass(slots=True)
class RenderResult:
    image: np.ndarray
    dirty_bbox: BBox | None = None
    previous_image: np.ndarray | None = None
    previous_error_sum: float | None = None
    incremental: bool = False


def _alpha_factor(config: Config, shape: Shape) -> float:
    if 0.0 <= config.fixed_alpha <= 1.0:
        return config.fixed_alpha
    return shape.alpha / 255.0


def _triangle_mask_local(
    tri: Triangle,
    min_x: int,
    min_y: int,
    max_x: int,
    max_y: int,
) -> np.ndarray:
    p0, p1, p2 = tri.points
    area = (int(p1[0]) - int(p0[0])) * (int(p2[1]) - int(p0[1])) - (
        int(p1[1]) - int(p0[1])
    ) * (int(p2[0]) - int(p0[0]))
    if area == 0:
        return np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=bool)

    x = np.arange(min_x, max_x + 1, dtype=np.int64)[None, :]
    y = np.arange(min_y, max_y + 1, dtype=np.int64)[:, None]
    edge0 = (x - p0[0]) * (p1[1] - p0[1]) - (y - p0[1]) * (p1[0] - p0[0])
    edge1 = (x - p1[0]) * (p2[1] - p1[1]) - (y - p1[1]) * (p2[0] - p1[0])
    edge2 = (x - p2[0]) * (p0[1] - p2[1]) - (y - p2[1]) * (p0[0] - p2[0])
    return ((edge0 >= 0) & (edge1 >= 0) & (edge2 >= 0)) | (
        (edge0 <= 0) & (edge1 <= 0) & (edge2 <= 0)
    )


def _circle_mask_local(
    circle: Circle,
    min_x: int,
    min_y: int,
    max_x: int,
    max_y: int,
) -> np.ndarray:
    x = np.arange(min_x, max_x + 1, dtype=np.int64)[None, :]
    y = np.arange(min_y, max_y + 1, dtype=np.int64)[:, None]
    cx, cy = circle.center
    return (x - cx) ** 2 + (y - cy) ** 2 <= circle.radius ** 2


def _shape_mask_local(
    shape: Shape,
    width: int,
    height: int,
    min_x: int,
    min_y: int,
    max_x: int,
    max_y: int,
) -> tuple[np.ndarray, bool]:
    cache_key = (width, height, min_x, min_y, max_x, max_y)
    cached = shape._mask_cache
    if cached is not None and cached[:6] == cache_key:
        return cached[6], cached[7]

    if isinstance(shape, Triangle):
        mask = _triangle_mask_local(shape, min_x, min_y, max_x, max_y)
    elif isinstance(shape, Circle):
        mask = _circle_mask_local(shape, min_x, min_y, max_x, max_y)
    else:
        mask = np.ones((max_y - min_y + 1, max_x - min_x + 1), dtype=bool)

    nonempty = bool(mask.any())
    mask.setflags(write=False)
    shape._mask_cache = (*cache_key, mask, nonempty)
    return mask, nonempty


def _blend_grayscale(
    region: np.ndarray,
    color: int,
    alpha: float,
    mask: np.ndarray | None,
) -> None:
    if alpha <= 0.0:
        return
    if alpha >= 1.0:
        if mask is None:
            region.fill(color)
        else:
            region[mask] = color
        return

    if mask is None:
        region *= 1.0 - alpha
        region += alpha * color
    else:
        values = region[mask]
        values *= 1.0 - alpha
        values += alpha * color
        region[mask] = values


def _blend_color(
    region: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
    mask: np.ndarray | None,
) -> None:
    if alpha <= 0.0:
        return
    if alpha >= 1.0:
        if mask is None:
            region[...] = color
        else:
            region[mask] = color
        return

    color_array = np.asarray(color, dtype=np.float32)
    if mask is None:
        region *= 1.0 - alpha
        region += alpha * color_array
    else:
        values = region[mask]
        values *= 1.0 - alpha
        values += alpha * color_array
        region[mask] = values


def _shape_bounds(shape: Shape, width: int, height: int) -> BBox:
    mask_cache = shape._mask_cache
    if mask_cache is not None and mask_cache[:2] == (width, height):
        return mask_cache[2:6]
    return shape.bounding_box(width, height)


def composite_shape(
    canvas: np.ndarray,
    shape: Shape,
    width: int,
    height: int,
    mode: ColorMode,
    config: Config,
) -> None:
    min_x, min_y, max_x, max_y = _shape_bounds(shape, width, height)
    if min_x > max_x or min_y > max_y:
        return

    mask = None
    if not isinstance(shape, Square):
        mask, nonempty = _shape_mask_local(shape, width, height, min_x, min_y, max_x, max_y)
        if not nonempty:
            return

    region = canvas[min_y : max_y + 1, min_x : max_x + 1]
    alpha = _alpha_factor(config, shape)
    if mode == "grayscale":
        assert isinstance(shape.color, int)
        _blend_grayscale(region, shape.color, alpha, mask)
    else:
        assert isinstance(shape.color, tuple)
        _blend_color(region, shape.color, alpha, mask)


def composite_shape_region(
    canvas: np.ndarray,
    shape: Shape,
    width: int,
    height: int,
    mode: ColorMode,
    config: Config,
    clip_bbox: BBox,
) -> None:
    shape_min_x, shape_min_y, shape_max_x, shape_max_y = _shape_bounds(shape, width, height)
    clip_min_x, clip_min_y, clip_max_x, clip_max_y = clip_bbox
    min_x = max(shape_min_x, clip_min_x)
    min_y = max(shape_min_y, clip_min_y)
    max_x = min(shape_max_x, clip_max_x)
    max_y = min(shape_max_y, clip_max_y)
    if min_x > max_x or min_y > max_y:
        return

    mask = None
    if not isinstance(shape, Square):
        full_mask, nonempty = _shape_mask_local(
            shape,
            width,
            height,
            shape_min_x,
            shape_min_y,
            shape_max_x,
            shape_max_y,
        )
        if not nonempty:
            return
        mask = full_mask[
            min_y - shape_min_y : max_y - shape_min_y + 1,
            min_x - shape_min_x : max_x - shape_min_x + 1,
        ]
        if not mask.any():
            return

    region = canvas[
        min_y - clip_min_y : max_y - clip_min_y + 1,
        min_x - clip_min_x : max_x - clip_min_x + 1,
    ]
    alpha = _alpha_factor(config, shape)
    if mode == "grayscale":
        assert isinstance(shape.color, int)
        _blend_grayscale(region, shape.color, alpha, mask)
    else:
        assert isinstance(shape.color, tuple)
        _blend_color(region, shape.color, alpha, mask)


def new_canvas(
    width: int,
    height: int,
    mode: ColorMode,
    background: int | tuple[int, int, int],
) -> np.ndarray:
    shape = (height, width) if mode == "grayscale" else (height, width, 3)
    canvas = np.empty(shape, dtype=np.float32)
    canvas[...] = background
    return canvas


def render_voronoi_float(
    individual: Individual,
    background: int | tuple[int, int, int],
) -> np.ndarray:
    width, height = individual.width, individual.height
    mode = individual.mode
    sites = [shape for shape in individual.shapes if isinstance(shape, VoronoiSite)]
    canvas = new_canvas(width, height, mode, background)
    if not sites:
        return canvas

    x = np.arange(width, dtype=np.float32)[None, :]
    y = np.arange(height, dtype=np.float32)[:, None]
    best_dist = np.full((height, width), np.inf, dtype=np.float32)
    for site in sites:
        sx, sy = site.point
        dist = (x - float(sx)) ** 2 + (y - float(sy)) ** 2
        mask = dist < best_dist
        canvas[mask] = site.color
        np.minimum(best_dist, dist, out=best_dist)

    return canvas


def render_voronoi(
    individual: Individual,
    background: int | tuple[int, int, int],
) -> np.ndarray:
    rendered = np.clip(render_voronoi_float(individual, background), 0, 255)
    return np.rint(rendered).astype(np.uint8)


def _render_cache_signature(
    individual: Individual,
    config: Config,
    background: int | tuple[int, int, int],
) -> tuple[object, ...]:
    return (
        individual.mode,
        background,
        config.fixed_alpha,
        config.shape_mode,
        config.renderer_backend,
        config.numba_min_pixels,
        config.cuda_min_pixels,
        config.compositing_cache_stride,
        config.compositing_cache_max_mb,
    )


def _render_incremental(
    individual: Individual,
    config: Config,
    background: int | tuple[int, int, int],
) -> RenderResult | None:
    if (
        not config.use_dirty_regions
        or not config.use_compositing_cache
        or individual._base_render is None
        or individual._dirty_from is None
        or individual._dirty_bbox is None
        or individual._compositing_cache_signature
        != _render_cache_signature(individual, config, background)
        or config.shape_mode == "voronoi"
        or any(isinstance(shape, VoronoiSite) for shape in individual.shapes)
    ):
        return None

    width, height = individual.width, individual.height
    base_render = individual._base_render
    if base_render.shape[:2] != (height, width):
        return None

    dirty_min_x, dirty_min_y, dirty_max_x, dirty_max_y = individual._dirty_bbox
    dirty_bbox = (
        max(0, dirty_min_x),
        max(0, dirty_min_y),
        min(width - 1, dirty_max_x),
        min(height - 1, dirty_max_y),
    )
    if dirty_bbox[0] > dirty_bbox[2] or dirty_bbox[1] > dirty_bbox[3]:
        return None

    cache = individual._compositing_cache
    prefix_index = -1
    max_prefix = min(individual._dirty_from - 1, len(cache) - 1)
    for index in range(max_prefix, -1, -1):
        if cache[index] is not None:
            prefix_index = index
            break

    min_x, min_y, max_x, max_y = dirty_bbox
    if prefix_index >= 0:
        prefix = cache[prefix_index]
        assert prefix is not None
        region = prefix[min_y : max_y + 1, min_x : max_x + 1].copy()
    else:
        region = new_canvas(max_x - min_x + 1, max_y - min_y + 1, individual.mode, background)

    backend = resolve_renderer_backend(config.renderer_backend, region.size, config)
    if backend == "numpy":
        for index in range(prefix_index + 1, len(individual.shapes)):
            composite_shape_region(
                region,
                individual.shapes[index],
                width,
                height,
                individual.mode,
                config,
                dirty_bbox,
            )
    else:
        try:
            region = render_shapes_accelerated(
                region,
                individual,
                config,
                prefix_index + 1,
                min_x,
                min_y,
                backend,
            )
        except Exception as exc:
            if config.renderer_backend != "auto":
                raise
            if mark_renderer_backend_failed(backend, exc):
                logging.warning("%s renderer failed; using NumPy: %s", backend, exc)
            if prefix_index >= 0:
                prefix = cache[prefix_index]
                assert prefix is not None
                region = prefix[min_y : max_y + 1, min_x : max_x + 1].copy()
            else:
                region = new_canvas(
                    max_x - min_x + 1,
                    max_y - min_y + 1,
                    individual.mode,
                    background,
                )
            for index in range(prefix_index + 1, len(individual.shapes)):
                composite_shape_region(
                    region,
                    individual.shapes[index],
                    width,
                    height,
                    individual.mode,
                    config,
                    dirty_bbox,
                )

    canvas = base_render.copy()
    canvas[min_y : max_y + 1, min_x : max_x + 1] = region
    if len(cache) < len(individual.shapes):
        cache.extend([None] * (len(individual.shapes) - len(cache)))
    elif len(cache) > len(individual.shapes):
        del cache[len(individual.shapes) :]
    if cache:
        cache[-1] = canvas

    previous_error_sum = individual._base_error_sum
    individual._compositing_cache = cache
    individual._clear_dirty_tracking()
    return RenderResult(
        image=canvas,
        dirty_bbox=dirty_bbox,
        previous_image=base_render,
        previous_error_sum=previous_error_sum,
        incremental=True,
    )


def _render_individual_full(
    individual: Individual,
    config: Config,
    background: int | tuple[int, int, int],
) -> np.ndarray:
    width, height = individual.width, individual.height
    mode = individual.mode
    if not config.use_compositing_cache and individual._compositing_cache:
        individual.clear_caches(include_shape_masks=False)
    cache_signature = _render_cache_signature(individual, config, background)
    if (
        config.use_compositing_cache
        and individual._compositing_cache_signature != cache_signature
    ):
        individual._compositing_cache = []
    cache = individual._compositing_cache if config.use_compositing_cache else []

    is_voronoi = config.shape_mode == "voronoi" or any(
        isinstance(shape, VoronoiSite) for shape in individual.shapes
    )
    if is_voronoi:
        final_index = len(individual.shapes) - 1
        if (
            cache
            and final_index >= 0
            and len(cache) > final_index
            and cache[final_index] is not None
        ):
            cached = cache[final_index]
            assert cached is not None
            return cached
        pixel_values = width * height * (1 if mode == "grayscale" else 3)
        backend = resolve_renderer_backend(config.renderer_backend, pixel_values, config)
        if backend == "numpy":
            canvas = render_voronoi_float(individual, background)
        else:
            try:
                canvas = render_voronoi_accelerated(
                    new_canvas(width, height, mode, background),
                    individual,
                    config,
                    backend,
                )
            except Exception as exc:
                if config.renderer_backend != "auto":
                    raise
                if mark_renderer_backend_failed(backend, exc):
                    logging.warning("%s renderer failed; using NumPy: %s", backend, exc)
                canvas = render_voronoi_float(individual, background)
        if config.use_compositing_cache:
            cache = [None] * len(individual.shapes)
            if final_index >= 0:
                cache[final_index] = canvas
            individual._compositing_cache = cache
            individual._compositing_cache_signature = cache_signature
        return canvas

    cached_index = -1
    if cache:
        max_cache_index = min(len(cache), len(individual.shapes)) - 1
        for index in range(max_cache_index, -1, -1):
            if cache[index] is not None:
                cached_index = index
                break

    if cached_index >= 0:
        cached = cache[cached_index]
        assert cached is not None
        if cached_index == len(individual.shapes) - 1:
            return cached
        canvas = cached.copy()
    else:
        canvas = new_canvas(width, height, mode, background)
        if config.use_compositing_cache:
            cache = [None] * len(individual.shapes)

    if config.use_compositing_cache:
        if len(cache) < len(individual.shapes):
            cache.extend([None] * (len(individual.shapes) - len(cache)))
        elif len(cache) > len(individual.shapes):
            del cache[len(individual.shapes) :]

    final_index = len(individual.shapes) - 1
    max_cache_bytes = int(config.compositing_cache_max_mb * 1024 * 1024)
    max_checkpoints = max(1, max_cache_bytes // max(1, canvas.nbytes))
    budget_stride = max(
        1,
        (len(individual.shapes) + max_checkpoints - 1) // max_checkpoints,
    )
    stride = max(config.compositing_cache_stride, budget_stride)
    backend = resolve_renderer_backend(config.renderer_backend, canvas.size, config)
    if backend == "numpy":
        for index in range(cached_index + 1, len(individual.shapes)):
            composite_shape(canvas, individual.shapes[index], width, height, mode, config)
            should_cache = index == final_index or (index + 1) % stride == 0
            if config.use_compositing_cache and should_cache:
                cache[index] = canvas if index == final_index else canvas.copy()
    elif cached_index < final_index:
        try:
            canvas = render_shapes_accelerated(
                canvas,
                individual,
                config,
                cached_index + 1,
                0,
                0,
                backend,
            )
            if config.use_compositing_cache and final_index >= 0:
                cache[final_index] = canvas
        except Exception as exc:
            if config.renderer_backend != "auto":
                raise
            if mark_renderer_backend_failed(backend, exc):
                logging.warning("%s renderer failed; using NumPy: %s", backend, exc)
            if cached_index >= 0:
                cached = cache[cached_index]
                assert cached is not None
                canvas = cached.copy()
            else:
                canvas = new_canvas(width, height, mode, background)
            for index in range(cached_index + 1, len(individual.shapes)):
                composite_shape(canvas, individual.shapes[index], width, height, mode, config)
                should_cache = index == final_index or (index + 1) % stride == 0
                if config.use_compositing_cache and should_cache:
                    cache[index] = canvas if index == final_index else canvas.copy()

    if config.use_compositing_cache:
        individual._compositing_cache = cache
        individual._compositing_cache_signature = cache_signature

    return canvas


def render_individual_for_scoring(
    individual: Individual,
    config: Config,
    background: int | tuple[int, int, int],
) -> RenderResult:
    incremental = _render_incremental(individual, config, background)
    if incremental is not None:
        return incremental
    image = _render_individual_full(individual, config, background)
    individual._clear_dirty_tracking()
    return RenderResult(image=image)


def render_individual_float(
    individual: Individual,
    config: Config,
    background: int | tuple[int, int, int],
) -> np.ndarray:
    return render_individual_for_scoring(individual, config, background).image


def render_individual(
    individual: Individual,
    config: Config,
    background: int | tuple[int, int, int],
) -> np.ndarray:
    rendered = np.clip(render_individual_float(individual, config, background), 0, 255)
    return np.rint(rendered).astype(np.uint8)
