from __future__ import annotations
import pandas as pd
import numpy as np

def wake_features(df5: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy-equivalent sleep/wake + WakeTime with vectorized ops.

    Requires 5-min indexed features with columns:
      ACC_Mean, ACC_Std, HR_Mean, HR_Std
    Returns (index-aligned):
      SleepWake, SleepWake1hr, SleepWake3hr, SleepWake4hr, WakeTime
    """
    out = df5.copy()

    # Ensure datetime index & sort
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[out.index.notna()].sort_index()

    # ---- Daily points system (vectorized; daywise means of original paper code) ----
    date_key = out.index.date
    out["_Date"] = date_key

    acc_m  = out.groupby("_Date")["ACC_Mean"].transform("mean")
    acc_s  = out.groupby("_Date")["ACC_Std"].transform("mean")
    hr_m   = out.groupby("_Date")["HR_Mean"].transform("mean")
    hr_s   = out.groupby("_Date")["HR_Std"].transform("mean")

    pts = ((out["ACC_Mean"] < acc_m).astype(int)
         + (out["ACC_Std"]  < acc_s).astype(int)
         + (out["HR_Mean"]  < hr_m).astype(int)
         + (out["HR_Std"]   < hr_s).astype(int))

    # WakeSleep = 0 if points>2, else 1; NaN if HR_Mean is NaN
    ws = np.where(out["HR_Mean"].isna(), np.nan, np.where(pts > 2, 0.0, 1.0))
    out["WakeSleep"] = ws

    # ---- Fixed-size rolling windows (12/24/36/48 samples @ 5-min) ----
    # min_periods=1 like legacy
    out["SleepWake"]    = out["WakeSleep"].rolling(24, min_periods=1).mean()  # 2h
    out["SleepWake1hr"] = out["WakeSleep"].rolling(12, min_periods=1).mean()  # 1h
    out["SleepWake3hr"] = out["WakeSleep"].rolling(36, min_periods=1).mean()  # 3h
    out["SleepWake4hr"] = out["WakeSleep"].rolling(48, min_periods=1).mean()  # 4h

    # ---- Daily WakeTime detection (same heuristic; efficient slicing) ----
    threshold = 0.1
    wake_rows = []

    # Small loop per-day (only ~8–10 per participant) – heavy lifting is vectorized!
    for day, g in out.groupby("_Date", group_keys=False):
        # Pre-noon coverage 
        pre_noon = g.between_time("00:00", "12:00")
        cnt = pre_noon["SleepWake3hr"].count()

        wakemin = np.nan
        if cnt >= 115:
            window = g.between_time("02:00", "10:00")
            if not window.empty:
                v = window["SleepWake3hr"].to_numpy()
                n = len(v)
                if n >= 16:
                    base = v[: n - 15]
                    f5   = v[5 : n - 10]
                    f15  = v[15: n]
                    mask = (base > threshold) & (base < f5) & (base < f15)
                    if mask.any():
                        i0 = int(mask.argmax())  # first True
                        t0 = window.index[i0]
                        wakemin = float(t0.hour * 60 + t0.minute)

        wake_rows.append((day, wakemin))

    wtdf = pd.DataFrame(wake_rows, columns=["_Date", "WakeTime"]).set_index("_Date")
    out = out.join(wtdf, on="_Date")

    # Fill remaining NaNs with the mean 
    if out["WakeTime"].isna().any():
        out["WakeTime"] = out["WakeTime"].fillna(out["WakeTime"].mean())

    # Return only the intended columns, index-aligned; drop helper
    out = out.drop(columns=["_Date"])
    return out[["WakeTime"]]
