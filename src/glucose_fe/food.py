# src/glucose_fe/food.py
from __future__ import annotations
import pandas as pd
from pathlib import Path

def food_features(food_csv: Path, base_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Aggregate Food_Log_<pid>.csv to 5-min cadence + rolling features.
    Returns only rolling aggregates and Eat flags (no raw macro columns)."""
    if not food_csv.exists():
        return pd.DataFrame(index=base_index)

    df = pd.read_csv(food_csv)

    # Normalize column names (handles stray spaces / case)
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
    )

    # Parse timestamp: prefer explicit time_begin, else date+time, else first column
    if "time_begin" in df.columns:
        t = pd.to_datetime(df["time_begin"], errors="coerce")
    elif {"date", "time"}.issubset(df.columns):
        t = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
    else:
        t = pd.to_datetime(df.iloc[:, 0], errors="coerce")

    df = df.assign(Time=t).dropna(subset=["Time"]).sort_values("Time")

    # Macros we use to build features
    macro_cols = ["calorie", "total_carb", "sugar", "protein"]
    for c in macro_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Sum per timestamp, then align to 5-min grid
    agg = (df.groupby("Time", as_index=True)[macro_cols]
             .sum()
             .sort_index())

    base = pd.DataFrame(index=base_index)
    x = base.join(agg.resample("5min").sum()).fillna(0)

    # Rolling sums over 2/8/24h (names match legacy)
    for hours in (2, 8, 24):
        w = f"{hours}h"
        for c in macro_cols:
            x[f"{c}{hours}hr"] = x[c].rolling(w, min_periods=1).sum()

    # Binary eat indicator + rollups
    x["Eat"] = (x["calorie"] > 0).astype(int)
    for hours in (2, 8, 24):
        w = f"{hours}h"
        x[f"Eatcnt{hours}hr"]  = x["Eat"].rolling(w, min_periods=1).sum()
        x[f"Eatmean{hours}hr"] = x["Eat"].rolling(w, min_periods=1).mean()

    # Drop instantaneous macro columns from the returned features
    x = x.drop(columns=macro_cols, errors="ignore")

    return x

