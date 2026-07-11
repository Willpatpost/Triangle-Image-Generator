# Genetic Shape Image Generator

Approximate an image by evolving primitive shapes with a genetic algorithm. The app now runs as a desktop GUI: choose an image from your computer, pick a shape family, choose RGB or B&W, watch the approximation evolve beside the target, and save the best result only when you want it.

## Quick Start

```bash
pip install -r requirements.txt
python tri_gen.py
```

## Workflow

1. Click **Choose Image** and select an image from your computer.
2. Choose a shape: triangle, circle, square, Voronoi, or mixed.
3. Choose `color` for RGB or `grayscale` for B&W.
4. Click **Start** to evolve the approximation.
5. Click **Stop** whenever you are happy with the result.
6. Click **Save Best** to write the current best image to a location you choose.

The app does not create iteration frames, comparison images, GIFs, or state files during normal use.

## Project Layout

```text
tri_gen.py          Desktop GUI entry point
core/               Algorithm, rendering, paths, and I/O
images/             Sample input image(s)
tests/              Unit tests
requirements.txt    Python dependencies
requirements-acceleration.txt  Optional Numba dependency
```

## Engine API

The approximation engine can run independently of the GUI. For best error per second, use a coarse-to-fine schedule and finish at downsample `1` for full resolution:

```python
from core import Config, run_progressive_evolution

image, fitness = run_progressive_evolution(
    "images/Yinyang.png",
    Config(mode="grayscale", shape_mode="triangle"),
    stages=((4, 2_000), (2, 2_000), (1, 4_000)),
    seed=42,
)
```

Each stage is `(downsample, iterations)`, ordered from a larger downsample to a smaller one. The engine scales the existing shape genome between stages instead of restarting.

The tuned defaults use residual-guided shape creation, sparse one-parameter mutations, exact float fitness, bounded compositing checkpoints, geometry-mask reuse, elite-only cache retention, and cache-preserving threaded evaluation for workloads large enough to benefit. Crossover remains available through `crossover_rate`, but defaults off because blending unrelated layers reduced convergence in deterministic benchmarks.

### Acceleration

Dirty-region rendering and scoring are enabled by default. A mutation records the smallest affected rectangle, rebuilds only that rectangle through subsequent layers, and updates the exact SSE total from the old and new rectangle errors.

The base install always uses NumPy. Install optional CPU JIT support with:

```bash
pip install -r requirements-acceleration.txt
```

Then inspect or select the renderer from Python:

```python
from core import Config, acceleration_status, available_renderer_backends

print(acceleration_status())
print(available_renderer_backends())

config = Config(renderer_backend="auto")   # cuda -> numba -> numpy by workload
config = Config(renderer_backend="numba")  # require Numba CPU JIT
config = Config(renderer_backend="cuda")   # require NVIDIA CUDA
```

For CUDA, install the toolkit-specific `numba-cuda` extra described in the [official installation guide](https://nvidia.github.io/numba-cuda/user/installation.html), such as `numba-cuda[cu12]` or `numba-cuda[cu13]`. Explicit unavailable backends raise a clear error; `auto` falls back to NumPy. `numba_min_pixels` and `cuda_min_pixels` control automatic selection thresholds.

## Requirements

- Python 3.10+
- Tkinter, usually bundled with Python
- See `requirements.txt`

## Tests

```bash
python -m unittest discover -v
```

## Tips

- Increase the **Preview size** value to use a smaller working image and evolve faster; lower it to retain more source detail.
- Use `grayscale` for faster B&W runs.
- `mixed` can make richer images but may need more time to settle.
