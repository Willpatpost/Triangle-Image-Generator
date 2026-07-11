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
```

## Requirements

- Python 3.10+
- Tkinter, usually bundled with Python
- See `requirements.txt`

## Tests

```bash
python -m unittest discover -v
```

## Tips

- Increase **Preview size** to make evolution faster on large images.
- Use `grayscale` for faster B&W runs.
- `mixed` can make richer images but may need more time to settle.
