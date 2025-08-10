from __future__ import annotations
import pandas as pd
import numpy as np
from scipy.signal import find_peaks


def eda_peak_counts(eda_df: pd.DataFrame, fs_hz: float = 4.0, base_freq: str = "5min") -> pd.DataFrame:
    """Detect EDA peaks on the raw 4 Hz signal, then count peaks per base window.
    Returns a DataFrame indexed by 5 min with columns PeakEDA, PeakEDA2hr_sum, PeakEDA2hr_mean.
    """
    s = eda_df[["Var"]].sort_index()
    # Ensure near-uniform sampling window-by-window is not required; use sample distance ~ 1s
    distance = max(int(fs_hz * 1.0), 1)
    peaks, _ = find_peaks(s["Var"].values, distance=distance, prominence=0.3)
    peak_index = s.index[peaks]
    peak_series = pd.Series(1, index=peak_index, name="Peak")
    # count peaks per base window
    peak_per_bin = peak_series.resample(base_freq).sum().fillna(0)
    df = pd.DataFrame({"PeakEDA": peak_per_bin})
    # 2h rolling on 5min cadence -> window="2h"
    df["PeakEDA2hr_sum"] = df["PeakEDA"].rolling("2h", min_periods=1).sum()
    df["PeakEDA2hr_mean"] = df["PeakEDA"].rolling("2h", min_periods=1).mean()
    return df