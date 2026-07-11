# Genetic Shape Image Generator

Approximate an image by evolving triangles, circles, squares, Voronoi cells, or a mixed set of primitive shapes. The desktop app shows the best evolution beside its target and writes nothing unless you explicitly save an image or session.

## Quick Start

```bash
pip install .
genetic-shape-image
```

You can also launch the app directly without installing it:

```bash
pip install -r requirements.txt
python app.py
```

## Desktop Workflow

1. Choose an image, shape family, RGB or B&W mode, and working resolution.
2. Tune the population, initial and maximum shape counts, renderer, or worker count when needed.
3. Start the evolution and stop it whenever you want to inspect or save the current best.
4. Continue the same run, or use **Refine** to carry its genome into a finer working resolution.
5. Save only the best image, or save the complete session and resume it later with **Open Session**.

`Full` working resolution uses every source pixel. Values such as `1/2` and `1/4` evolve a smaller raster for faster iteration. Refining scales the existing genome instead of restarting it.

Session files are versioned JSON. Current files preserve the best genome, active population, configuration, stagnation state, random-generator state, source path, and source hash. Older files using the original `triangles` and `min_triangles` keys remain loadable.

## Project Layout

```text
app.py                         Desktop GUI entry point
core/                          Evolution, rendering, fitness, shapes, state, and I/O
images/                        Sample input images
tests/                         Unit and integration tests
.github/workflows/ci.yml       Windows/Linux test matrix
pyproject.toml                 Package metadata and GUI entry point
requirements.txt               Base dependencies
requirements-acceleration.txt  Optional Numba dependency
```

## Engine API

The engine is independent of the GUI. For strong error reduction per second, evolve from coarse to fine and finish at downsample `1`:

```python
from core import Config, run_progressive_evolution

image, fitness = run_progressive_evolution(
    "images/Yinyang.png",
    Config(mode="grayscale", shape_mode="triangle"),
    stages=((4, 2_000), (2, 2_000), (1, 4_000)),
    seed=42,
)
```

Each stage is `(downsample, iterations)`, ordered from a larger downsample to a smaller one. The engine scales the shape genome between stages instead of restarting.

The tuned defaults use residual-guided creation, sparse one-parameter mutations, exact float fitness, dirty-region incremental scoring, bounded compositing checkpoints, geometry-mask reuse, elite-only cache retention, and cache-preserving threaded evaluation. Crossover remains available through `crossover_rate`, but defaults off because blending unrelated layers reduced convergence in deterministic benchmarks.

## Acceleration

Dirty-region rendering and exact incremental SSE scoring are enabled by default. A mutation records the smallest affected rectangle, rebuilds that rectangle through subsequent layers, and replaces only its old contribution to total error.

The base install always supports NumPy. Install optional CPU JIT support with either command:

```bash
pip install ".[acceleration]"
# or
pip install -r requirements-acceleration.txt
```

Inspect or select a backend from Python:

```python
from core import Config, acceleration_status, available_renderer_backends

print(acceleration_status())
print(available_renderer_backends())

config = Config(renderer_backend="auto")   # workload-aware fallback
config = Config(renderer_backend="numba")  # require Numba CPU JIT
config = Config(renderer_backend="cuda")   # require NVIDIA CUDA
```

For CUDA, install the toolkit-specific `numba-cuda` extra described in the [official installation guide](https://nvidia.github.io/numba-cuda/user/installation.html), such as `numba-cuda[cu12]` or `numba-cuda[cu13]`. Explicit unavailable backends raise a clear error; `auto` falls back safely. `numba_min_pixels` and `cuda_min_pixels` control automatic selection thresholds.

## Development

The project supports Python 3.10 and newer. Tkinter is normally included with Python but may be a separate operating-system package on Linux.

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

Build distributable wheel and source archives:

```bash
pip install ".[dev]"
python -m build
```

GitHub Actions validates the package and test suite on Windows and Linux across Python 3.10, 3.12, and 3.14.
