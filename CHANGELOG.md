# Changelog

## 2.3.0 — 2026-10-14

First release under the `genetic-shape-image-generator` package name, aligned
with the project's own 2.3 lineage.

- MIT license; package metadata now names the author and license.
- Evolution engine in `core` with triangle, circle, square, Voronoi, and mixed
  genomes.
- Tkinter desktop app (`app.py`) with start/stop/continue/refine, session
  save/open, and best-image export.
- Progressive coarse-to-fine evolution via `run_progressive_evolution`.
- Dirty-region incremental rendering with exact incremental fitness scoring.
- Optional Numba CPU and CUDA renderer backends with automatic fallback.
- Versioned JSON session files (format version 2) with atomic writes, source
  hash verification, and legacy version-1 support.
- `run_evolution` now delegates to `EvolutionSession` and raises
  `FileNotFoundError` when a resume file is missing.
- CI runs the test suite against the installed package instead of the source
  tree.
- Sample image `images/Spongebob LeBron.png` removed (third-party art that is
  not redistributable under the project license).
