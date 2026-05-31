from __future__ import annotations

import unittest
from contextlib import redirect_stderr
from io import StringIO

from tri_gen import main


class CliValidationTests(unittest.TestCase):
    def assert_cli_error(self, *args: str) -> None:
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as context:
                main(list(args))
        self.assertNotEqual(context.exception.code, 0)

    def test_rejects_invalid_population_values(self) -> None:
        self.assert_cli_error("--pop-size", "1")
        self.assert_cli_error("--elite", "0")
        self.assert_cli_error("--pop-size", "5", "--elite", "5")

    def test_rejects_invalid_iteration_and_shape_values(self) -> None:
        self.assert_cli_error("--iterations", "0")
        self.assert_cli_error("--triangles", "0")
        self.assert_cli_error("--triangles", "10", "--max-triangles", "9")
        self.assert_cli_error("--triangles", "4", "--max-triangles", "4")
        self.assert_cli_error("--workers", "-1")


if __name__ == "__main__":
    unittest.main()
