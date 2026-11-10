"""Empirical split conformal quantile calibration; no exchangeability claim."""
import numpy as np


def cqr_offset(y, lower, upper, coverage=0.8):
    """Return finite-sample conformal order statistic fitted on calibration only."""
    y, lower, upper = (np.asarray(x, dtype=float) for x in (y, lower, upper))
    if not 0 < coverage < 1 or y.ndim != 1 or y.size == 0 or y.shape != lower.shape or y.shape != upper.shape:
        raise ValueError("invalid calibration data or coverage")
    if not all(np.isfinite(x).all() for x in (y, lower, upper)):
        raise ValueError("calibration data must be finite")
    if np.any(lower > upper):
        raise ValueError("lower exceeds upper")
    scores = np.maximum(lower - y, y - upper)
    rank = int(np.ceil((len(y) + 1) * coverage))
    if rank > len(y):
        raise ValueError("not enough calibration samples for requested coverage")
    return float(np.partition(scores, rank - 1)[rank - 1])


def calibrated_interval(lower, median, upper, offset, rated_power=2.05):
    """Apply calibration and enforce physical limits and quantile ordering."""
    lo, mid, hi = np.broadcast_arrays(*[np.asarray(x, dtype=float) for x in (lower, median, upper)])
    if rated_power <= -0.05 or not np.isfinite(rated_power) or not np.isfinite(offset) or not all(np.isfinite(x).all() for x in (lo, mid, hi)):
        raise ValueError("non-finite values or invalid rated power")
    mid = np.clip(mid, -0.05, rated_power)
    lo = np.minimum(np.clip(lo - offset, -0.05, rated_power), mid)
    hi = np.maximum(np.clip(hi + offset, -0.05, rated_power), mid)
    return lo, mid, hi
