"""Shared utilities for time standardization and baseline construction."""

from typing import Any

import numpy as np


def standardized_time_index(length: int, xp: Any = np) -> Any:
    """Return standardized time index with zero mean and unit variance."""
    t = xp.arange(length, dtype=np.float32)
    return (t - xp.mean(t)) / (xp.std(t) + 1e-6)


def linear_baseline(intercept: Any, slope: Any, t_std: Any) -> Any:
    """Return linear baseline defined by intercept + slope * standardized time."""
    return intercept + slope * t_std
