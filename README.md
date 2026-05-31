# Genetic Algorithm for Image Approximation Using Shapes

Approximate any image by evolving **primitive shapes** with a genetic algorithm. Supports triangles, circles, squares, Voronoi cells, mixed-shape runs, RGB color by default, optional grayscale, parallel fitness evaluation, save/resume, evolution GIFs, and side-by-side comparison output.

## Quick Start

```bash
pip install -r requirements.txt

python tri_gen.py --downsample 2
```

The sample image `images/Yinyang.png` is used by default. Results are written to `output/`.

## Output Files

| File | Description |
|------|-------------|
| `output/final.png` | Best approximation |
| `output/comparison.png` | Target \| result \| diff |
| `output/iteration*.png` | Snapshots when fitness improves |
| `output/evolution.gif` | Animated evolution |
| `output/best_state.json` | Resumable saved state |

## CLI

```bash
python tri_gen.py [options]
python tri_gen.py --help
```

### Input

| Option | Default | Description |
|--------|---------|-------------|
| `--image`, `-i` | `images/Yinyang.png` | Target image (filename also searches `images/`) |
| `--mode` | `color` | `color` or `grayscale` |
| `--shape` | `triangle` | `triangle`, `circle`, `square`, `voronoi`, or `mixed` |
| `--downsample` | `1` | Shrink image before evolving |

### Evolution

| Option | Default | Description |
|--------|---------|-------------|
| `--iterations` | `30000` | Max generations |
| `--resume` | `None` | Continue from `output/best_state.json` |
| `--seed` | `None` | Random seed |

### Population

| Option | Default | Description |
|--------|---------|-------------|
| `--pop-size` | `30` | Candidates per generation |
| `--elite` | `6` | Top candidates kept each generation |
| `--triangles`, `--shapes` | `15` | Starting shape count per candidate |
| `--max-triangles`, `--max-shapes` | `150` | Shape cap per candidate |

### Output

| Option | Default | Description |
|--------|---------|-------------|
| `--output`, `-o` | `output` | Results directory |
| `--no-comparison` | off | Skip comparison image |
| `--quiet` | off | Suppress logs |
| `--workers` | `0` | Parallel fitness workers (`0` = CPU count minus one) |

### Examples

```bash
# Default sample image, faster preview
python tri_gen.py --downsample 2 --iterations 5000

# Image by filename (looks in images/)
python tri_gen.py -i Yinyang.png

# Your own image anywhere on disk
python tri_gen.py -i C:\Photos\portrait.jpg --downsample 4

# Color mode, custom output folder
python tri_gen.py -i photo.jpg --mode color -o output/color-run

# Mixed shape run
python tri_gen.py -i photo.jpg --shape mixed --mode color

# Voronoi cells
python tri_gen.py -i photo.jpg --shape voronoi --shapes 80

# Resume previous run
python tri_gen.py --resume
python tri_gen.py --resume output/best_state.json --iterations 20000
```

## Project Layout

```text
tri_gen.py          CLI entry point
core/               Algorithm, rendering, paths, and I/O
images/             Sample input image(s)
output/             Generated results (created on run)
requirements.txt    Python dependencies
```

## Requirements

- Python 3.10+
- See `requirements.txt`

## Tests

```bash
python -m unittest discover -v
```

## Tips

- Start with `--downsample 2` or `4` for quick previews.
- Use `--mode grayscale` for faster monochrome runs.
- Use `--resume` to continue from `output/best_state.json`.
- Color mode is slower and usually needs more iterations.
