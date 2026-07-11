#!/usr/bin/env python3
"""Desktop GUI for evolving shapes into an image."""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from core.config import Config
from core.evolver import EvolutionSession, EvolutionSnapshot
from core.io import save_image


CANVAS_WIDTH = 520
CANVAS_HEIGHT = 430


@dataclass
class WorkerUpdate:
    run_id: int
    snapshot: EvolutionSnapshot
    rendered: np.ndarray


@dataclass
class WorkerError:
    run_id: int
    error: Exception


class TriangleImageApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Triangle Image Generator")
        self.minsize(1120, 680)

        self.image_path: Path | None = None
        self.target_photo: ImageTk.PhotoImage | None = None
        self.evolution_photo: ImageTk.PhotoImage | None = None
        self.best_image: np.ndarray | None = None
        self.session: EvolutionSession | None = None
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.updates: queue.Queue[WorkerUpdate | WorkerError] = queue.Queue()
        self.run_id = 0

        self.shape_var = tk.StringVar(value="triangle")
        self.mode_var = tk.StringVar(value="color")
        self.downsample_var = tk.IntVar(value=2)
        self.status_var = tk.StringVar(value="Choose an image to begin.")

        self._build_ui()
        self.after(80, self._poll_updates)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.configure("Panel.TFrame", padding=12)
        style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 10))

        toolbar = ttk.Frame(self, padding=(16, 14, 16, 8))
        toolbar.pack(fill=tk.X)

        ttk.Button(toolbar, text="Choose Image", command=self._choose_image).pack(side=tk.LEFT)

        ttk.Label(toolbar, text="Shape").pack(side=tk.LEFT, padx=(18, 6))
        shape_box = ttk.Combobox(
            toolbar,
            textvariable=self.shape_var,
            values=("triangle", "circle", "square", "voronoi", "mixed"),
            width=12,
            state="readonly",
        )
        shape_box.pack(side=tk.LEFT)

        ttk.Label(toolbar, text="Mode").pack(side=tk.LEFT, padx=(18, 6))
        mode_box = ttk.Combobox(
            toolbar,
            textvariable=self.mode_var,
            values=("color", "grayscale"),
            width=10,
            state="readonly",
        )
        mode_box.pack(side=tk.LEFT)

        ttk.Label(toolbar, text="Preview size").pack(side=tk.LEFT, padx=(18, 6))
        ttk.Spinbox(
            toolbar,
            from_=1,
            to=6,
            textvariable=self.downsample_var,
            width=4,
            justify=tk.CENTER,
        ).pack(side=tk.LEFT)

        self.start_button = ttk.Button(toolbar, text="Start", command=self._start_evolving, state=tk.DISABLED)
        self.start_button.pack(side=tk.LEFT, padx=(24, 6))

        self.stop_button = ttk.Button(toolbar, text="Stop", command=self._stop_evolving, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT)

        self.save_button = ttk.Button(toolbar, text="Save Best", command=self._save_best, state=tk.DISABLED)
        self.save_button.pack(side=tk.RIGHT)

        body = ttk.Frame(self, padding=(16, 8, 16, 16))
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(1, weight=1)

        ttk.Label(body, text="Evolution", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(body, text="Target", style="Title.TLabel").grid(row=0, column=1, sticky="w", padx=(12, 0))

        self.evolution_label = ttk.Label(body, anchor=tk.CENTER, background="#151515")
        self.evolution_label.grid(row=1, column=0, sticky="nsew", pady=(8, 0), padx=(0, 6))

        self.target_label = ttk.Label(body, anchor=tk.CENTER, background="#151515")
        self.target_label.grid(row=1, column=1, sticky="nsew", pady=(8, 0), padx=(6, 0))

        status = ttk.Label(self, textvariable=self.status_var, style="Status.TLabel", padding=(16, 0, 16, 14))
        status.pack(fill=tk.X)

    def _choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=(
                ("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"),
                ("All files", "*.*"),
            ),
        )
        if not path:
            return

        self._stop_evolving()
        self.image_path = Path(path)
        self.best_image = None
        self.session = None
        self.save_button.configure(state=tk.DISABLED)
        self.start_button.configure(state=tk.NORMAL)
        self._show_target(self.image_path)
        self.evolution_label.configure(image="", text="")
        self.status_var.set(f"Ready: {self.image_path.name}")

    def _start_evolving(self) -> None:
        if self.image_path is None:
            messagebox.showinfo("Choose an image", "Choose an image before starting.")
            return
        if self.worker and self.worker.is_alive():
            return

        assert self.image_path is not None
        self.run_id += 1
        run_id = self.run_id
        image_path = self.image_path
        self.stop_event.clear()
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.save_button.configure(state=tk.DISABLED)
        self.status_var.set("Starting evolution...")

        mode = self.mode_var.get()
        shape = self.shape_var.get()
        downsample = max(1, self.downsample_var.get())
        self.worker = threading.Thread(
            target=self._run_worker,
            args=(run_id, image_path, mode, shape, downsample),
            daemon=True,
        )
        self.worker.start()

    def _run_worker(self, run_id: int, image_path: Path, mode: str, shape: str, downsample: int) -> None:
        try:
            config = Config(
                mode=mode,  # type: ignore[arg-type]
                shape_mode=shape,  # type: ignore[arg-type]
                pop_size=24,
                nb_elite=5,
                nb_elements_initial=18,
                nb_elements_max=140,
                min_triangles=5,
                hill_climb_interval=75,
                hill_climb_attempts=1,
                max_workers=1,
                use_compositing_cache=True,
                enable_logging=False,
            )
            session = EvolutionSession(str(image_path), config, downsample=downsample)
            self.session = session

            last_render = 0.0
            while not self.stop_event.is_set():
                snapshot = session.step_many(4)
                now = time.monotonic()
                if snapshot.improved or now - last_render >= 0.25:
                    self.updates.put(WorkerUpdate(run_id=run_id, snapshot=snapshot, rendered=session.render_best()))
                    last_render = now
        except Exception as exc:
            self.updates.put(WorkerError(run_id=run_id, error=exc))

    def _poll_updates(self) -> None:
        try:
            while True:
                update = self.updates.get_nowait()
                if isinstance(update, WorkerError):
                    if update.run_id != self.run_id:
                        continue
                    self._stop_evolving()
                    messagebox.showerror("Evolution failed", str(update.error))
                    self.status_var.set("Evolution stopped after an error.")
                    continue
                if update.run_id != self.run_id:
                    continue
                self.best_image = update.rendered
                self._show_evolution(update.rendered)
                self.save_button.configure(state=tk.NORMAL)
                self.status_var.set(
                    "Iteration {0.iteration} | Best {0.best_fitness:.6f} | "
                    "Current {0.current_fitness:.6f} | Shapes {0.shape_count}".format(update.snapshot)
                )
        except queue.Empty:
            pass
        finally:
            if self.worker and not self.worker.is_alive():
                self.start_button.configure(state=tk.NORMAL if self.image_path else tk.DISABLED)
                self.stop_button.configure(state=tk.DISABLED)
            self.after(80, self._poll_updates)

    def _stop_evolving(self) -> None:
        self.stop_event.set()
        self.stop_button.configure(state=tk.DISABLED)
        if self.image_path is not None:
            self.start_button.configure(state=tk.NORMAL)

    def _save_best(self) -> None:
        if self.best_image is None:
            return
        mode = "color" if self.best_image.ndim == 3 else "grayscale"
        path = filedialog.asksaveasfilename(
            title="Save best image",
            defaultextension=".png",
            filetypes=(("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("All files", "*.*")),
        )
        if not path:
            return
        save_image(self.best_image, path, mode)
        self.status_var.set(f"Saved {Path(path).name}")

    def _show_target(self, path: Path) -> None:
        image = Image.open(path).convert("RGB")
        self.target_photo = self._photo_for_panel(image)
        self.target_label.configure(image=self.target_photo, text="")

    def _show_evolution(self, array: np.ndarray) -> None:
        if array.ndim == 2:
            image = Image.fromarray(array, mode="L").convert("RGB")
        else:
            image = Image.fromarray(array, mode="RGB")
        self.evolution_photo = self._photo_for_panel(image)
        self.evolution_label.configure(image=self.evolution_photo, text="")

    def _photo_for_panel(self, image: Image.Image) -> ImageTk.PhotoImage:
        image = image.copy()
        image.thumbnail((CANVAS_WIDTH, CANVAS_HEIGHT), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image)

    def _on_close(self) -> None:
        self._stop_evolving()
        self.destroy()


def main() -> int:
    app = TriangleImageApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
