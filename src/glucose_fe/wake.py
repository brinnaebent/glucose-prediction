# src/glucose_fe/wake.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Optional

def wake_features(
    df5: pd.DataFrame,
    *,
    base_freq: str = "5min",
    # DEFAULT UPDATED: 04:00–14:00
    search_window: Optional[Tuple[str, str]] = ("04:00", "14:00"),
    threshold: float = 0.10,          # min SleepWake3hr to consider a rise
    sustain_1: str = "25min",         # must stay higher for at least this
    sustain_2: str = "75min",         # and mostly higher over this
    sustain_frac_long: float = 0.80,  # ≥80% of long window higher than baseline
    eps: float = 0.0,                 # margin above baseline
    # Minimum valid samples inside the search window (5-min cadence → 24 ≈ 2h)
    min_window_count: int = 24
) -> pd.DataFrame:
    """
    Returns columns:
      WakeSleep, SleepWake, SleepWake1hr, SleepWake3hr, SleepWake4hr, WakeTime
    WakeTime units = minutes after midnight.

    Behavior:
      - Only searches within `search_window` (default 04:00–14:00).
      - Requires at least `min_window_count` non-NaN samples of SleepWake3hr inside the window.
      - If no valid candidate or insufficient coverage → WakeTime = NaN.
      - No unbounded fallback, no filling.
    """
    out = df5.copy()

    # Ensure datetime index & sort
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[out.index.notna()].sort_index()

    need = {"ACC_Mean", "ACC_Std", "HR_Mean", "HR_Std"}
    if not need.issubset(out.columns):
        return pd.DataFrame(index=out.index, columns=[
            "WakeSleep","SleepWake","SleepWake1hr","SleepWake3hr","SleepWake4hr","WakeTime"
        ])

    # ---- Daily points → WakeSleep (vectorized) ----
    out["_Date"] = out.index.date
    acc_m = out.groupby("_Date")["ACC_Mean"].transform("mean")
    acc_s = out.groupby("_Date")["ACC_Std"].transform("mean")
    hr_m  = out.groupby("_Date")["HR_Mean"].transform("mean")
    hr_s  = out.groupby("_Date")["HR_Std"].transform("mean")

    pts = ((out["ACC_Mean"] < acc_m).astype(int)
         + (out["ACC_Std"]  < acc_s).astype(int)
         + (out["HR_Mean"]  < hr_m).astype(int)
         + (out["HR_Std"]   < hr_s).astype(int))

    # semantics: points>2 ⇒ sleep(0), else wake(1); NaN if HR_Mean is NaN
    out["WakeSleep"] = np.where(out["HR_Mean"].isna(), np.nan,
                         np.where(pts > 2, 0.0, 1.0))

    # ---- Rolling means (1/2/3/4h) on 5-min cadence (or whatever base_freq implies) ----
    step = pd.to_timedelta(base_freq)
    per_hour = max(1, int(pd.Timedelta("1h") / step))
    out["SleepWake"]    = out["WakeSleep"].rolling(2 * per_hour, min_periods=1).mean()  # 2h
    out["SleepWake1hr"] = out["WakeSleep"].rolling(1 * per_hour, min_periods=1).mean()  # 1h
    out["SleepWake3hr"] = out["WakeSleep"].rolling(3 * per_hour, min_periods=1).mean()  # 3h
    out["SleepWake4hr"] = out["WakeSleep"].rolling(4 * per_hour, min_periods=1).mean()  # 4h

    # sustained checks in samples (5-min cadence ⇒ 25min≈5, 75min≈15)
    n1 = max(1, int(pd.to_timedelta(sustain_1) / step))
    n2 = max(n1 + 1, int(pd.to_timedelta(sustain_2) / step))

    def _bounded_candidate(g: pd.DataFrame) -> Optional[pd.Timestamp]:
        if search_window is None:
            return None
        lo, hi = search_window
        w = g.between_time(lo, hi)
        # require minimum coverage inside the window
        if w["SleepWake3hr"].count() < min_window_count:
            return None
        v = w["SleepWake3hr"].to_numpy()
        idx = w.index
        # find first sustained rise
        for i in range(0, len(v) - n2):
            base = v[i]
            if base <= threshold:
                continue
            short_ok = np.all(v[i+1:i+1+n1] > base + eps)
            long_ok  = (np.sum(v[i+1:i+1+n2] > base + eps) >= int(np.ceil(sustain_frac_long * n2)))
            if short_ok and long_ok:
                return idx[i]
        return None

    # per-day detection (bounded only; no fallback, no fill)
    wake_rows = []
    for day, g in out.groupby("_Date", group_keys=False):
        t = _bounded_candidate(g)
        wakemin = float(t.hour * 60 + t.minute) if isinstance(t, pd.Timestamp) else np.nan
        wake_rows.append((day, wakemin))

    wtdf = pd.DataFrame(wake_rows, columns=["_Date", "WakeTime"]).set_index("_Date")
    out = out.join(wtdf, on="_Date")

    out = out.drop(columns=["_Date"])
    return out[["WakeTime"]]


