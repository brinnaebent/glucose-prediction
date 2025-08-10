from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Dict

@dataclass(frozen=True)
class FeatureWindows:
    base_freq: str = "5min"
    base_feature_window: str = "10min"

# ---- basic rolling stats for a single-channel sensor ----

def rolling_stats(var_series: pd.Series, window: str) -> pd.DataFrame:
    # Ensure DateTimeIndex
    s = var_series.sort_index()
    # Align to base frequency if not already
    s = s.resample("5min").mean()
    roll = s.rolling(window, min_periods=1)
    df = pd.DataFrame({
        "Mean": roll.mean(),
        "Std": roll.std(),
        "Min": roll.min(),
        "Max": roll.max(),
        "Q1G": roll.quantile(0.25),
        "Q3G": roll.quantile(0.75),
        "Skew": roll.skew(),
    })
    return df


def compute_sensor_feature_frames(sensors: Dict[str, pd.DataFrame], fw: FeatureWindows) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for name, df in sensors.items():
        if "Var" not in df.columns:
            raise ValueError(f"Sensor '{name}' missing 'Var' column")
        feats = rolling_stats(df["Var"], window=fw.base_feature_window)
        feats.columns = [f"{name}_{c}" for c in feats.columns]
        out[name] = feats
    return out


def merge_on_index(frames: Dict[str, pd.DataFrame], base_freq: str = "5min") -> pd.DataFrame:
    # Build a union time index at the requested base cadence
    all_idx = None
    for df in frames.values():
        idx = df.index
        if all_idx is None:
            all_idx = idx
        else:
            all_idx = all_idx.union(idx)
    if all_idx is None:
        return pd.DataFrame()
    grid = pd.date_range(all_idx.min(), all_idx.max(), freq=base_freq)
    merged = pd.DataFrame(index=grid)
    for name, df in frames.items():
        merged = merged.join(df.reindex(grid))
    return merged