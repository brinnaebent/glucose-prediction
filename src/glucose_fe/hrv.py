from __future__ import annotations
import numpy as np
import pandas as pd


def _diff_ms(ibi_s: pd.Series) -> np.ndarray:
    ibi_ms = ibi_s.values * 1000.0
    return np.diff(ibi_ms)


def rmssd(ibi_s: pd.Series) -> float:
    d = np.abs(_diff_ms(ibi_s))
    if d.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(d * d)))


def sdnn(ibi_s: pd.Series) -> float:
    v = np.var(ibi_s.values * 1000.0, ddof=1) if len(ibi_s) > 1 else np.nan
    return float(np.sqrt(v)) if not np.isnan(v) else np.nan


def pnnx(ibi_s: pd.Series, x_ms: float = 50.0) -> tuple[float, float]:
    d = np.abs(_diff_ms(ibi_s))
    if d.size == 0:
        return (np.nan, np.nan)
    n = float((d > x_ms).sum())
    p = float(100.0 * n / d.size)
    return (p, n)


def hrv_window_features(ibi_df: pd.DataFrame, freq: str = "5min", x_ms: float = 50.0) -> pd.DataFrame:
    """Compute HRV features per time window using IBI series with a DateTimeIndex."""
    groups = ibi_df[["IBI"]].sort_index().groupby(pd.Grouper(freq=freq))
    rows = []
    for t, g in groups:
        if g.empty:
            continue
        ibi_s = g["IBI"]
        rm = rmssd(ibi_s)
        sd = sdnn(ibi_s)
        p, n = pnnx(ibi_s, x_ms=x_ms)
        rows.append({
            "Time": t,
            "maxHrv": float(np.max(ibi_s.values * 1000.0)),
            "minHrv": float(np.min(ibi_s.values * 1000.0)),
            "meanHrv": float(np.mean(ibi_s.values * 1000.0)),
            "medianHrv": float(np.median(ibi_s.values * 1000.0)),
            "sdnn": sd,
            "nnx": n,
            "pnnx": p,
            "rmssd": rm,
        })
    out = pd.DataFrame(rows).set_index("Time").sort_index()
    return out