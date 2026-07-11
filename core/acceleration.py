from __future__ import annotations

import importlib.util
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np

from core.config import Config, RendererBackend
from core.shapes import Circle, Individual, Square, Triangle, VoronoiSite


ResolvedBackend = Literal["numpy", "numba", "cuda"]


class AcceleratorUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AccelerationStatus:
    numba_available: bool
    cuda_available: bool
    numba_error: str | None = None
    cuda_error: str | None = None


@dataclass(slots=True)
class EncodedShapes:
    kinds: np.ndarray
    geometry: np.ndarray
    bounds: np.ndarray
    colors: np.ndarray
    alphas: np.ndarray


_BACKEND_FAILURES: dict[ResolvedBackend, str] = {}
_BACKEND_FAILURE_LOCK = threading.Lock()


def mark_renderer_backend_failed(backend: ResolvedBackend, error: Exception) -> bool:
    if backend == "numpy":
        return False
    with _BACKEND_FAILURE_LOCK:
        if backend in _BACKEND_FAILURES:
            return False
        _BACKEND_FAILURES[backend] = str(error)
        return True


def _backend_failure(backend: ResolvedBackend) -> str | None:
    with _BACKEND_FAILURE_LOCK:
        return _BACKEND_FAILURES.get(backend)


def _numba_installed() -> bool:
    return importlib.util.find_spec("numba") is not None


@lru_cache(maxsize=1)
def _numba_status() -> tuple[bool, str | None]:
    if not _numba_installed():
        return False, "Numba is not installed"
    try:
        import numba  # noqa: F401

        return True, None
    except Exception as exc:
        return False, str(exc)


@lru_cache(maxsize=1)
def _cuda_status() -> tuple[bool, str | None]:
    numba_available, numba_error = _numba_status()
    if not numba_available:
        return False, numba_error
    try:
        from numba import cuda

        if cuda.is_available():
            return True, None
        return False, "No compatible CUDA device or driver was detected"
    except Exception as exc:
        return False, str(exc)


def acceleration_status() -> AccelerationStatus:
    numba_available, numba_error = _numba_status()
    cuda_available, cuda_error = _cuda_status()
    numba_failure = _backend_failure("numba")
    cuda_failure = _backend_failure("cuda")
    if numba_failure is not None:
        numba_available, numba_error = False, numba_failure
    if cuda_failure is not None:
        cuda_available, cuda_error = False, cuda_failure
    return AccelerationStatus(
        numba_available=numba_available,
        cuda_available=cuda_available,
        numba_error=numba_error,
        cuda_error=cuda_error,
    )


def available_renderer_backends() -> tuple[ResolvedBackend, ...]:
    status = acceleration_status()
    backends: list[ResolvedBackend] = ["numpy"]
    if status.numba_available:
        backends.append("numba")
    if status.cuda_available:
        backends.append("cuda")
    return tuple(backends)


def resolve_renderer_backend(
    requested: RendererBackend,
    pixel_values: int,
    config: Config,
) -> ResolvedBackend:
    if requested == "numpy":
        return "numpy"
    if requested == "numba":
        failure = _backend_failure("numba")
        if failure is not None:
            raise AcceleratorUnavailable("The Numba renderer failed earlier: " + failure)
        available, error = _numba_status()
        if not available:
            raise AcceleratorUnavailable(
                "The Numba renderer was requested, but Numba is unavailable. "
                "Install requirements-acceleration.txt or use renderer_backend='auto'. "
                f"Details: {error or 'unknown error'}"
            )
        return "numba"
    if requested == "cuda":
        failure = _backend_failure("cuda")
        if failure is not None:
            raise AcceleratorUnavailable("The CUDA renderer failed earlier: " + failure)
        available, error = _cuda_status()
        if not available:
            raise AcceleratorUnavailable(
                "The CUDA renderer was requested, but CUDA is unavailable: " + (error or "unknown error")
            )
        return "cuda"

    if pixel_values >= config.cuda_min_pixels and _backend_failure("cuda") is None:
        available, _ = _cuda_status()
        if available:
            return "cuda"
    if pixel_values >= config.numba_min_pixels and _backend_failure("numba") is None:
        available, _ = _numba_status()
        if available:
            return "numba"
    return "numpy"


def encode_shapes(individual: Individual, config: Config) -> EncodedShapes:
    count = len(individual.triangles)
    kinds = np.empty(count, dtype=np.int8)
    geometry = np.zeros((count, 6), dtype=np.int32)
    bounds = np.empty((count, 4), dtype=np.int32)
    colors = np.zeros((count, 3), dtype=np.float32)
    alphas = np.empty(count, dtype=np.float32)

    for index, shape in enumerate(individual.triangles):
        if isinstance(shape, Triangle):
            kinds[index] = 0
            geometry[index] = shape.points.reshape(6)
        elif isinstance(shape, Circle):
            kinds[index] = 1
            geometry[index, :3] = (shape.center[0], shape.center[1], shape.radius)
        elif isinstance(shape, Square):
            kinds[index] = 2
            geometry[index, :3] = (shape.top_left[0], shape.top_left[1], shape.side)
        else:
            kinds[index] = 3
            geometry[index, :2] = shape.point

        bounds[index] = shape.bounding_box(individual.width, individual.height)
        if isinstance(shape.color, int):
            colors[index, 0] = shape.color
        else:
            colors[index] = shape.color
        alphas[index] = (
            config.fixed_alpha
            if 0.0 <= config.fixed_alpha <= 1.0
            else shape.alpha / 255.0
        )

    return EncodedShapes(kinds, geometry, bounds, colors, alphas)


def _render_gray_impl(
    canvas: np.ndarray,
    kinds: np.ndarray,
    geometry: np.ndarray,
    bounds: np.ndarray,
    colors: np.ndarray,
    alphas: np.ndarray,
    start_index: int,
    origin_x: int,
    origin_y: int,
) -> None:
    global_max_x = origin_x + canvas.shape[1] - 1
    global_max_y = origin_y + canvas.shape[0] - 1
    for shape_index in range(start_index, kinds.shape[0]):
        min_x = max(origin_x, bounds[shape_index, 0])
        min_y = max(origin_y, bounds[shape_index, 1])
        max_x = min(global_max_x, bounds[shape_index, 2])
        max_y = min(global_max_y, bounds[shape_index, 3])
        if min_x > max_x or min_y > max_y:
            continue
        alpha = alphas[shape_index]
        inverse_alpha = np.float32(1.0) - alpha
        color = colors[shape_index, 0]
        for y in range(min_y, max_y + 1):
            local_y = y - origin_y
            for x in range(min_x, max_x + 1):
                kind = kinds[shape_index]
                inside = False
                if kind == 0:
                    x0 = geometry[shape_index, 0]
                    y0 = geometry[shape_index, 1]
                    x1 = geometry[shape_index, 2]
                    y1 = geometry[shape_index, 3]
                    x2 = geometry[shape_index, 4]
                    y2 = geometry[shape_index, 5]
                    area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
                    if area != 0:
                        edge0 = (x - x0) * (y1 - y0) - (y - y0) * (x1 - x0)
                        edge1 = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
                        edge2 = (x - x2) * (y0 - y2) - (y - y2) * (x0 - x2)
                        inside = (
                            edge0 >= 0 and edge1 >= 0 and edge2 >= 0
                        ) or (
                            edge0 <= 0 and edge1 <= 0 and edge2 <= 0
                        )
                elif kind == 1:
                    dx = x - geometry[shape_index, 0]
                    dy = y - geometry[shape_index, 1]
                    radius = geometry[shape_index, 2]
                    inside = dx * dx + dy * dy <= radius * radius
                elif kind == 2:
                    left = geometry[shape_index, 0]
                    top = geometry[shape_index, 1]
                    side = geometry[shape_index, 2]
                    inside = x >= left and x <= left + side and y >= top and y <= top + side
                if inside:
                    local_x = x - origin_x
                    if alpha >= 1.0:
                        canvas[local_y, local_x] = color
                    elif alpha > 0.0:
                        canvas[local_y, local_x] *= inverse_alpha
                        canvas[local_y, local_x] += alpha * color


def _render_color_impl(
    canvas: np.ndarray,
    kinds: np.ndarray,
    geometry: np.ndarray,
    bounds: np.ndarray,
    colors: np.ndarray,
    alphas: np.ndarray,
    start_index: int,
    origin_x: int,
    origin_y: int,
) -> None:
    global_max_x = origin_x + canvas.shape[1] - 1
    global_max_y = origin_y + canvas.shape[0] - 1
    for shape_index in range(start_index, kinds.shape[0]):
        min_x = max(origin_x, bounds[shape_index, 0])
        min_y = max(origin_y, bounds[shape_index, 1])
        max_x = min(global_max_x, bounds[shape_index, 2])
        max_y = min(global_max_y, bounds[shape_index, 3])
        if min_x > max_x or min_y > max_y:
            continue
        alpha = alphas[shape_index]
        inverse_alpha = np.float32(1.0) - alpha
        for y in range(min_y, max_y + 1):
            local_y = y - origin_y
            for x in range(min_x, max_x + 1):
                kind = kinds[shape_index]
                inside = False
                if kind == 0:
                    x0 = geometry[shape_index, 0]
                    y0 = geometry[shape_index, 1]
                    x1 = geometry[shape_index, 2]
                    y1 = geometry[shape_index, 3]
                    x2 = geometry[shape_index, 4]
                    y2 = geometry[shape_index, 5]
                    area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
                    if area != 0:
                        edge0 = (x - x0) * (y1 - y0) - (y - y0) * (x1 - x0)
                        edge1 = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
                        edge2 = (x - x2) * (y0 - y2) - (y - y2) * (x0 - x2)
                        inside = (
                            edge0 >= 0 and edge1 >= 0 and edge2 >= 0
                        ) or (
                            edge0 <= 0 and edge1 <= 0 and edge2 <= 0
                        )
                elif kind == 1:
                    dx = x - geometry[shape_index, 0]
                    dy = y - geometry[shape_index, 1]
                    radius = geometry[shape_index, 2]
                    inside = dx * dx + dy * dy <= radius * radius
                elif kind == 2:
                    left = geometry[shape_index, 0]
                    top = geometry[shape_index, 1]
                    side = geometry[shape_index, 2]
                    inside = x >= left and x <= left + side and y >= top and y <= top + side
                if inside:
                    local_x = x - origin_x
                    for channel in range(3):
                        if alpha >= 1.0:
                            canvas[local_y, local_x, channel] = colors[shape_index, channel]
                        elif alpha > 0.0:
                            canvas[local_y, local_x, channel] *= inverse_alpha
                            canvas[local_y, local_x, channel] += (
                                alpha * colors[shape_index, channel]
                            )


def _render_voronoi_gray_impl(
    canvas: np.ndarray,
    geometry: np.ndarray,
    colors: np.ndarray,
) -> None:
    for y in range(canvas.shape[0]):
        for x in range(canvas.shape[1]):
            best_distance = np.float64(np.inf)
            best_color = canvas[y, x]
            for site_index in range(geometry.shape[0]):
                dx = x - geometry[site_index, 0]
                dy = y - geometry[site_index, 1]
                distance = dx * dx + dy * dy
                if distance < best_distance:
                    best_distance = distance
                    best_color = colors[site_index, 0]
            canvas[y, x] = best_color


def _render_voronoi_color_impl(
    canvas: np.ndarray,
    geometry: np.ndarray,
    colors: np.ndarray,
) -> None:
    for y in range(canvas.shape[0]):
        for x in range(canvas.shape[1]):
            best_distance = np.float64(np.inf)
            best_index = -1
            for site_index in range(geometry.shape[0]):
                dx = x - geometry[site_index, 0]
                dy = y - geometry[site_index, 1]
                distance = dx * dx + dy * dy
                if distance < best_distance:
                    best_distance = distance
                    best_index = site_index
            if best_index >= 0:
                for channel in range(3):
                    canvas[y, x, channel] = colors[best_index, channel]


_NUMBA_FUNCTIONS: tuple[object, object, object, object] | None = None
_NUMBA_LOCK = threading.Lock()


def _load_numba_functions() -> tuple[object, object, object, object]:
    global _NUMBA_FUNCTIONS
    if _NUMBA_FUNCTIONS is not None:
        return _NUMBA_FUNCTIONS
    with _NUMBA_LOCK:
        if _NUMBA_FUNCTIONS is not None:
            return _NUMBA_FUNCTIONS
        try:
            from numba import njit
        except Exception as exc:
            raise AcceleratorUnavailable("Numba could not be imported") from exc

        _NUMBA_FUNCTIONS = (
            njit(cache=True, nogil=True)(_render_gray_impl),
            njit(cache=True, nogil=True)(_render_color_impl),
            njit(cache=True, nogil=True)(_render_voronoi_gray_impl),
            njit(cache=True, nogil=True)(_render_voronoi_color_impl),
        )
    return _NUMBA_FUNCTIONS


_CUDA_KERNELS: tuple[object, object, object, object] | None = None
_CUDA_LOCK = threading.Lock()


def _load_cuda_kernels() -> tuple[object, object, object, object]:
    global _CUDA_KERNELS
    if _CUDA_KERNELS is not None:
        return _CUDA_KERNELS
    with _CUDA_LOCK:
        if _CUDA_KERNELS is not None:
            return _CUDA_KERNELS
        kernels = _build_cuda_kernels()
        _CUDA_KERNELS = kernels
        return kernels


def _build_cuda_kernels() -> tuple[object, object, object, object]:
    try:
        from numba import cuda
    except Exception as exc:
        raise AcceleratorUnavailable("Numba CUDA could not be imported") from exc

    @cuda.jit(device=True)
    def inside_shape(kind, geometry, shape_index, x, y):
        if kind == 0:
            x0 = geometry[shape_index, 0]
            y0 = geometry[shape_index, 1]
            x1 = geometry[shape_index, 2]
            y1 = geometry[shape_index, 3]
            x2 = geometry[shape_index, 4]
            y2 = geometry[shape_index, 5]
            area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
            if area == 0:
                return False
            edge0 = (x - x0) * (y1 - y0) - (y - y0) * (x1 - x0)
            edge1 = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
            edge2 = (x - x2) * (y0 - y2) - (y - y2) * (x0 - x2)
            return (
                edge0 >= 0 and edge1 >= 0 and edge2 >= 0
            ) or (
                edge0 <= 0 and edge1 <= 0 and edge2 <= 0
            )
        if kind == 1:
            dx = x - geometry[shape_index, 0]
            dy = y - geometry[shape_index, 1]
            radius = geometry[shape_index, 2]
            return dx * dx + dy * dy <= radius * radius
        if kind == 2:
            left = geometry[shape_index, 0]
            top = geometry[shape_index, 1]
            side = geometry[shape_index, 2]
            return x >= left and x <= left + side and y >= top and y <= top + side
        return False

    @cuda.jit
    def render_gray_kernel(canvas, kinds, geometry, colors, alphas, start_index, origin_x, origin_y):
        x, y = cuda.grid(2)
        if y >= canvas.shape[0] or x >= canvas.shape[1]:
            return
        global_x = x + origin_x
        global_y = y + origin_y
        value = canvas[y, x]
        for shape_index in range(start_index, kinds.shape[0]):
            if inside_shape(kinds[shape_index], geometry, shape_index, global_x, global_y):
                alpha = alphas[shape_index]
                if alpha >= 1.0:
                    value = colors[shape_index, 0]
                elif alpha > 0.0:
                    value *= 1.0 - alpha
                    value += alpha * colors[shape_index, 0]
        canvas[y, x] = value

    @cuda.jit
    def render_color_kernel(canvas, kinds, geometry, colors, alphas, start_index, origin_x, origin_y):
        x, y = cuda.grid(2)
        if y >= canvas.shape[0] or x >= canvas.shape[1]:
            return
        global_x = x + origin_x
        global_y = y + origin_y
        red = canvas[y, x, 0]
        green = canvas[y, x, 1]
        blue = canvas[y, x, 2]
        for shape_index in range(start_index, kinds.shape[0]):
            if inside_shape(kinds[shape_index], geometry, shape_index, global_x, global_y):
                alpha = alphas[shape_index]
                if alpha >= 1.0:
                    red = colors[shape_index, 0]
                    green = colors[shape_index, 1]
                    blue = colors[shape_index, 2]
                elif alpha > 0.0:
                    inverse_alpha = 1.0 - alpha
                    red = red * inverse_alpha + alpha * colors[shape_index, 0]
                    green = green * inverse_alpha + alpha * colors[shape_index, 1]
                    blue = blue * inverse_alpha + alpha * colors[shape_index, 2]
        canvas[y, x, 0] = red
        canvas[y, x, 1] = green
        canvas[y, x, 2] = blue

    @cuda.jit
    def voronoi_gray_kernel(canvas, geometry, colors):
        x, y = cuda.grid(2)
        if y >= canvas.shape[0] or x >= canvas.shape[1]:
            return
        best_distance = 1.0e30
        best_color = canvas[y, x]
        for site_index in range(geometry.shape[0]):
            dx = x - geometry[site_index, 0]
            dy = y - geometry[site_index, 1]
            distance = dx * dx + dy * dy
            if distance < best_distance:
                best_distance = distance
                best_color = colors[site_index, 0]
        canvas[y, x] = best_color

    @cuda.jit
    def voronoi_color_kernel(canvas, geometry, colors):
        x, y = cuda.grid(2)
        if y >= canvas.shape[0] or x >= canvas.shape[1]:
            return
        best_distance = 1.0e30
        best_index = -1
        for site_index in range(geometry.shape[0]):
            dx = x - geometry[site_index, 0]
            dy = y - geometry[site_index, 1]
            distance = dx * dx + dy * dy
            if distance < best_distance:
                best_distance = distance
                best_index = site_index
        if best_index >= 0:
            canvas[y, x, 0] = colors[best_index, 0]
            canvas[y, x, 1] = colors[best_index, 1]
            canvas[y, x, 2] = colors[best_index, 2]

    return (
        render_gray_kernel,
        render_color_kernel,
        voronoi_gray_kernel,
        voronoi_color_kernel,
    )


def _render_cuda(
    canvas: np.ndarray,
    encoded: EncodedShapes,
    start_index: int,
    origin_x: int,
    origin_y: int,
) -> np.ndarray:
    from numba import cuda

    gray_kernel, color_kernel, _, _ = _load_cuda_kernels()
    device_canvas = cuda.to_device(np.ascontiguousarray(canvas))
    device_kinds = cuda.to_device(encoded.kinds)
    device_geometry = cuda.to_device(encoded.geometry)
    device_colors = cuda.to_device(encoded.colors)
    device_alphas = cuda.to_device(encoded.alphas)
    threads = (16, 16)
    blocks = (
        (canvas.shape[1] + threads[0] - 1) // threads[0],
        (canvas.shape[0] + threads[1] - 1) // threads[1],
    )
    kernel = gray_kernel if canvas.ndim == 2 else color_kernel
    kernel[blocks, threads](
        device_canvas,
        device_kinds,
        device_geometry,
        device_colors,
        device_alphas,
        start_index,
        origin_x,
        origin_y,
    )
    return device_canvas.copy_to_host()


def render_shapes_accelerated(
    canvas: np.ndarray,
    individual: Individual,
    config: Config,
    start_index: int,
    origin_x: int,
    origin_y: int,
    backend: ResolvedBackend,
) -> np.ndarray:
    encoded = encode_shapes(individual, config)
    if backend == "numba":
        gray, color, _, _ = _load_numba_functions()
        function = gray if canvas.ndim == 2 else color
        function(
            canvas,
            encoded.kinds,
            encoded.geometry,
            encoded.bounds,
            encoded.colors,
            encoded.alphas,
            start_index,
            origin_x,
            origin_y,
        )
        return canvas
    if backend == "cuda":
        return _render_cuda(canvas, encoded, start_index, origin_x, origin_y)
    raise ValueError("render_shapes_accelerated requires numba or cuda")


def render_voronoi_accelerated(
    canvas: np.ndarray,
    individual: Individual,
    config: Config,
    backend: ResolvedBackend,
) -> np.ndarray:
    encoded = encode_shapes(individual, config)
    site_mask = encoded.kinds == 3
    geometry = np.ascontiguousarray(encoded.geometry[site_mask])
    colors = np.ascontiguousarray(encoded.colors[site_mask])
    if geometry.shape[0] == 0:
        return canvas

    if backend == "numba":
        _, _, gray, color = _load_numba_functions()
        function = gray if canvas.ndim == 2 else color
        function(canvas, geometry, colors)
        return canvas
    if backend == "cuda":
        from numba import cuda

        _, _, gray_kernel, color_kernel = _load_cuda_kernels()
        device_canvas = cuda.to_device(np.ascontiguousarray(canvas))
        device_geometry = cuda.to_device(geometry)
        device_colors = cuda.to_device(colors)
        threads = (16, 16)
        blocks = (
            (canvas.shape[1] + threads[0] - 1) // threads[0],
            (canvas.shape[0] + threads[1] - 1) // threads[1],
        )
        kernel = gray_kernel if canvas.ndim == 2 else color_kernel
        kernel[blocks, threads](device_canvas, device_geometry, device_colors)
        return device_canvas.copy_to_host()
    raise ValueError("render_voronoi_accelerated requires numba or cuda")
