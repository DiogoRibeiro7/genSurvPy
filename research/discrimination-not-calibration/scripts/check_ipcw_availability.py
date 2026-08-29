"""Compatibility entry point for the pre-freeze IPCW availability gate."""

from __future__ import annotations

from audit_ipcw_availability import main

if __name__ == "__main__":
    raise SystemExit(main())
