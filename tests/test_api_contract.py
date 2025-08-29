from __future__ import annotations

import inspect

import gen_surv


def test_generate_signature_stable() -> None:
    sig = inspect.signature(gen_surv.generate)
    assert "model:" in str(sig) and "**kwargs" in str(sig)
