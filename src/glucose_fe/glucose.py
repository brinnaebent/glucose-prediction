from __future__ import annotations
import pandas as pd


def resample_glucose_5min(glucose_df: pd.DataFrame) -> pd.DataFrame:
    """Dexcom glucose is already sampled ~every 5 minutes.
    Normalize to an exact 5-minute grid and interpolate small gaps so all
    other sensors can align to this base cadence.
    """
    g = glucose_df[["Glucose"]].sort_index()
    g5 = g.resample("5min").mean().interpolate("time", limit_direction="both")
    return g5