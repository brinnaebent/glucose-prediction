from __future__ import annotations
import polars as pl
from typing import Dict

# Resample to exact 5-min cadence first (mean), then 10-min rolling on that grid

def _to_5min(df: pl.DataFrame) -> pl.DataFrame:
    # expects columns: Time (Datetime), Var (Float)
    return (
        df.sort("Time")
          .group_by_dynamic("Time", every="5m", period="5m", closed="right")
          .agg(pl.col("Var").mean().alias("Var"))
    )


def _rolling_10min(df5: pl.DataFrame, prefix: str) -> pl.DataFrame:
    # 10-minute window = 2 bins of 5 minutes
    return (
        df5.sort("Time")
           .with_columns([
               pl.col("Var").rolling_mean(window_size=2, min_periods=1).alias(f"{prefix}_Mean"),
               pl.col("Var").rolling_std(window_size=2, ddof=1, min_periods=1).alias(f"{prefix}_Std"),
               pl.col("Var").rolling_min(window_size=2, min_periods=1).alias(f"{prefix}_Min"),
               pl.col("Var").rolling_max(window_size=2, min_periods=1).alias(f"{prefix}_Max"),
               pl.col("Var").rolling_quantile(0.25, window_size=2, min_periods=1).alias(f"{prefix}_Q1G"),
               pl.col("Var").rolling_quantile(0.75, window_size=2, min_periods=1).alias(f"{prefix}_Q3G"),
               pl.col("Var").rolling_skew(window_size=2).alias(f"{prefix}_Skew"),
           ])
           .drop("Var")
    )


def compute_sensor_feature_frames_polars(sensors: Dict[str, "pl.DataFrame"]) -> Dict[str, "pl.DataFrame"]:
    out: Dict[str, pl.DataFrame] = {}
    for name, df in sensors.items():
        if not {"Time", "Var"}.issubset(df.columns):
            raise ValueError(f"Sensor '{name}' must have Time/Var columns for polars path")
        df5 = _to_5min(df)
        out[name] = _rolling_10min(df5, prefix=name)
    return out


def merge_on_time_polars(frames: Dict[str, "pl.DataFrame"], base_times: "pl.DataFrame") -> "pl.DataFrame":
    merged = base_times
    for df in frames.values():
        merged = merged.join(df, on="Time", how="left")
    return merged