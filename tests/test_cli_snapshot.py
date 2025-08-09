from __future__ import annotations

import subprocess
import sys


def run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "gen_surv", *args],
        text=True,
        capture_output=True,
    )


def test_cli_help_shows_usage() -> None:
    cp = run_cli(["--help"])
    assert cp.returncode == 0, cp.stderr
    assert "Generate synthetic survival datasets." in cp.stdout
