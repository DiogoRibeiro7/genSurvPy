"""Put the study package on the path for these tests.

The research code is not installed -- it is not part of the ``gen_surv``
distribution and must never become an install-time dependency of it -- so the
tests add its ``src`` directory themselves. Nothing outside this directory
imports ``survival_misspec``.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
