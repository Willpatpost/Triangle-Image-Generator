from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from core.config import Config
from core.evolver import EvolutionSnapshot
from core.io import EvolutionState
from core.shapes import Individual
from gui_state import (
    EvolutionController,
    SessionPreview,
    SessionSettings,
    WorkerError,
    WorkerFinished,
    WorkerSessionReady,
    WorkerUpdate,
    compute_control_states,
)


def _settings(**overrides) -> SessionSettings:
    values = dict(
        image_path=Path("target.png"),
        mode="grayscale",
        shape_mode="mixed",
        downsample=2,
        population=4,
        initial_shapes=2,
        max_shapes=8,
        renderer_backend="numpy",
        max_workers=1,
    )
    values.update(overrides)
    return SessionSettings(**values)


class StubSession:
    def __init__(self, stop_event, stop_after_batches=2, downsample=4):
        self._stop_event = stop_event
        self._stop_after = stop_after_batches
        self.downsample = downsample
        self.batches = 0
        self.iteration = 0
        self.refined_to = None
        self.global_best = SimpleNamespace(fitness=0.5, shapes=[1, 2, 3])
        self.ranked = [(0, 0.6)]

    def step_many(self, count: int) -> EvolutionSnapshot:
        self.batches += 1
        self.iteration += count
        improved = self.batches == 1
        if self.batches >= self._stop_after:
            self._stop_event.set()
        return EvolutionSnapshot(
            iteration=self.iteration,
            best_fitness=0.5,
            current_fitness=0.6,
            shape_count=3,
            improved=improved,
        )

    def render_best(self) -> np.ndarray:
        return np.zeros((4, 4), dtype=np.uint8)

    def refine_resolution(self, downsample: int) -> None:
        self.refined_to = downsample
        self.downsample = downsample


class StubFactory:
    def __init__(self, session) -> None:
        self.session = session

    def create(self, settings, config):
        return self.session

    def resume(self, resume_path, settings):
        return self.session


class ControlStatesTests(unittest.TestCase):
    def test_running_disables_everything_but_stop(self) -> None:
        states = compute_control_states(
            running=True,
            image_chosen=True,
            has_session=True,
            has_resume=False,
            has_best=True,
            session_downsample=2,
        )
        self.assertTrue(states.stop)
        self.assertIsNone(states.start_text)
        for flag in (
            states.choose_image,
            states.open_session,
            states.settings,
            states.start_enabled,
            states.refine,
            states.save_best,
            states.save_session,
        ):
            self.assertFalse(flag)

    def test_idle_states_track_session_resume_and_best(self) -> None:
        no_image = compute_control_states(
            running=False,
            image_chosen=False,
            has_session=False,
            has_resume=False,
            has_best=False,
            session_downsample=None,
        )
        self.assertFalse(no_image.start_enabled)
        self.assertEqual(no_image.start_text, "Start")
        self.assertFalse(no_image.save_best)
        self.assertFalse(no_image.save_session)
        self.assertFalse(no_image.refine)

        ready = compute_control_states(
            running=False,
            image_chosen=True,
            has_session=False,
            has_resume=False,
            has_best=False,
            session_downsample=None,
        )
        self.assertTrue(ready.start_enabled)
        self.assertEqual(ready.start_text, "Start")

        resumed = compute_control_states(
            running=False,
            image_chosen=True,
            has_session=False,
            has_resume=True,
            has_best=True,
            session_downsample=None,
        )
        self.assertEqual(resumed.start_text, "Resume")
        self.assertTrue(resumed.save_best)
        self.assertFalse(resumed.save_session)

        continued = compute_control_states(
            running=False,
            image_chosen=True,
            has_session=True,
            has_resume=False,
            has_best=True,
            session_downsample=2,
        )
        self.assertEqual(continued.start_text, "Continue")
        self.assertTrue(continued.save_session)
        self.assertTrue(continued.refine)

        full_scale = compute_control_states(
            running=False,
            image_chosen=True,
            has_session=True,
            has_resume=False,
            has_best=True,
            session_downsample=1,
        )
        self.assertFalse(full_scale.refine)


class ControllerTests(unittest.TestCase):
    def test_worker_lifecycle_delivers_messages_in_order(self) -> None:
        controller = EvolutionController()
        stub = StubSession(controller.stop_event)
        settings = _settings()
        run_id = controller.begin_run()

        controller.run_worker(run_id, settings, None, stub, None, session_factory=StubFactory(stub))
        messages = controller.drain()

        self.assertIsInstance(messages[0], WorkerSessionReady)
        self.assertIsInstance(messages[-2], WorkerUpdate)
        self.assertIsInstance(messages[-1], WorkerFinished)
        self.assertFalse(messages[-1].failed)
        self.assertTrue(any(isinstance(message, WorkerUpdate) for message in messages))
        self.assertIs(controller.session, stub)
        self.assertEqual(controller.session_settings, settings)
        self.assertIsNone(controller.resume_path)
        self.assertEqual(stub.batches, 2)

    def test_stale_run_messages_are_dropped(self) -> None:
        controller = EvolutionController()
        controller.run_id = 7
        stale_snapshot = EvolutionSnapshot(
            iteration=1, best_fitness=0.5, current_fitness=0.6, shape_count=3, improved=False
        )
        controller.updates.put(
            WorkerUpdate(run_id=6, snapshot=stale_snapshot, rendered=np.zeros((2, 2), dtype=np.uint8))
        )
        controller.updates.put(WorkerFinished(run_id=7, failed=False))

        messages = controller.drain()

        self.assertEqual(len(messages), 1)
        self.assertIsInstance(messages[0], WorkerFinished)

    def test_worker_error_path_reports_failure(self) -> None:
        controller = EvolutionController()

        def raise_step(count: int) -> EvolutionSnapshot:
            raise RuntimeError("boom")

        stub = StubSession(controller.stop_event)
        stub.step_many = raise_step
        run_id = controller.begin_run()

        controller.run_worker(run_id, _settings(), None, stub, None, session_factory=StubFactory(stub))
        messages = controller.drain()

        self.assertIsInstance(messages[0], WorkerSessionReady)
        self.assertIsInstance(messages[1], WorkerError)
        self.assertIsInstance(messages[-1], WorkerFinished)
        self.assertTrue(messages[-1].failed)

    def test_refine_updates_downsample_in_ready_message(self) -> None:
        controller = EvolutionController()
        stub = StubSession(controller.stop_event, downsample=4)
        run_id = controller.begin_run()

        controller.run_worker(
            run_id,
            _settings(downsample=4),
            None,
            stub,
            None,
            refine_to=2,
            session_factory=StubFactory(stub),
        )
        messages = controller.drain()

        ready = messages[0]
        assert isinstance(ready, WorkerSessionReady)
        self.assertEqual(stub.refined_to, 2)
        self.assertEqual(ready.settings.downsample, 2)

    def test_begin_run_increments_and_clears_stop(self) -> None:
        controller = EvolutionController()
        controller.request_stop()
        first = controller.begin_run()
        second = controller.begin_run()
        self.assertEqual(second, first + 1)
        self.assertFalse(controller.stop_event.is_set())

    def test_start_worker_runs_until_the_stub_stops(self) -> None:
        controller = EvolutionController()
        stub = StubSession(controller.stop_event)

        controller.start_worker(_settings(), None, session_factory=StubFactory(stub))
        controller.join_worker(timeout=5.0)

        self.assertFalse(controller.is_running())
        messages = controller.drain()
        self.assertIsInstance(messages[0], WorkerSessionReady)
        self.assertIsInstance(messages[-1], WorkerFinished)

    def test_reset_for_settings_change_clears_resumable_state(self) -> None:
        controller = EvolutionController()
        controller.session = StubSession(controller.stop_event)
        controller.resume_path = Path("session.json")
        self.assertTrue(controller.has_resumable_state())

        controller.reset_for_settings_change()

        self.assertFalse(controller.has_resumable_state())
        self.assertIsNone(controller.session)
        self.assertIsNone(controller.resume_path)

    def test_session_preview_sets_resume_path(self) -> None:
        controller = EvolutionController()
        controller.run_id = 3
        state = EvolutionState(
            iteration=1,
            best_fitness=0.4,
            best=Individual(width=2, height=2, mode="grayscale", shapes=[]),
            config=Config(),
            background=255,
        )
        session_path = Path("session.json")
        controller.updates.put(
            SessionPreview(
                run_id=3,
                state=state,
                source=Path("target.png"),
                session_path=session_path,
                rendered=np.zeros((2, 2), dtype=np.uint8),
            )
        )

        messages = controller.drain()

        self.assertEqual(len(messages), 1)
        self.assertEqual(controller.resume_path, session_path)
        self.assertIsNone(controller.session)


if __name__ == "__main__":
    unittest.main()
