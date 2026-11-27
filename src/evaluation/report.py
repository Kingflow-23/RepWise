"""Helpers shared by the suites: latency summaries and the two output files."""

import json
import math
import statistics
from datetime import datetime
from pathlib import Path

from config import evaluation_config as config

from .metrics.common import Ratio


def median(values: list[float | None]) -> float | None:
    """Median of the measured values, or ``None`` when nothing was measured."""
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (``q`` in [0, 1]), or ``None`` for no values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def now() -> str:
    """Run timestamp stored in every report."""
    return datetime.now().isoformat(timespec="seconds")


def ms(value: float | None) -> str:
    """Format a latency for Markdown tables."""
    return "—" if value is None else f"{value:.1f} ms"


def write_run(suite: str, payload: dict, markdown: str, label: str = "") -> Path:
    """Write ``report.md`` and ``results.json`` to a new run folder.

    The folder is ``outputs/eval_runs/<timestamp>_<suite>[_<label>]``.
    ``Ratio`` values are stored as ``{"value", "num", "den"}``.

    Returns:
        The run folder.
    """
    name = f"{datetime.now():%Y%m%d-%H%M%S}_{suite}" + (f"_{label}" if label else "")
    run_dir = config.OUT_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_to_json),
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(markdown, encoding="utf-8")
    return run_dir


def _to_json(value):
    if isinstance(value, Ratio):
        return value.to_dict()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")
