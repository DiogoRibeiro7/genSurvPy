from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "gen_surv", *args],
        text=True,
        capture_output=True,
        check=True,
    )


def test_cli_help_snapshot() -> None:
    cp = run_cli(["--help"])
    out = cp.stdout
    golden = Path(__file__).parent / "baselines" / "cli_help.txt"
    if "--update-baselines" in sys.argv:
        golden.write_text(out, encoding="utf-8")
        return
    assert out == golden.read_text(encoding="utf-8")
