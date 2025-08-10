# src/glucose_fe/io.py  (PATCH)

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import re
from typing import Literal

Sensor = Literal["EDA", "TEMP", "HR", "ACC", "Dexcom", "IBI", "BVP"]

@dataclass(frozen=True)
class ParticipantPaths:
    base: Path          
    participant_id: str  

    def sensor_path(self, sensor: Sensor) -> Path:
        # data/<pid>/<Sensor>_<pid>.csv
        return self.base / self.participant_id / f"{sensor}_{self.participant_id}.csv"

    def food_log_path(self) -> Path:
        # data/<pid>/Food_Log_<pid>.csv
        return self.base / self.participant_id / f"Food_Log_{self.participant_id}.csv"

# ---------- helpers ----------

def _pd_read_csv_fast(path: Path, **kwargs) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, engine="pyarrow", **kwargs)
    except Exception:
        df = pd.read_csv(path, **kwargs)
    df.columns = df.columns.astype(str).str.strip()
    return df


def _to_time_index(df: pd.DataFrame, time_col_names=("datetime", "Time", "Timestamp")) -> pd.DataFrame:
    # pick the first matching time column (case-insensitive)
    lower = {c.lower(): c for c in df.columns}
    tkey = next((nm for nm in time_col_names if nm.lower() in lower), None)
    if tkey is None:
        # if not above, assume first column is time
        tkey = df.columns[0]
    tcol = lower.get(tkey.lower(), tkey)
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol]).sort_values(tcol).set_index(tcol)
    df.index = df.index.tz_localize(None)
    df.index.name = "Time"
    return df

def _find_time_col(df):
    for k in ("datetime","time","timestamp","date","Time"):
        if k in df.columns: return k
    return df.columns[0]

def _to_time_index_smart(df, prefer_formats=None):
    tcol = _find_time_col(df)
    s = None
    if prefer_formats:
        for fmt in prefer_formats:
            cand = pd.to_datetime(df[tcol], format=fmt, errors="coerce")
            if cand.notna().mean() >= 0.95:  # accept if most rows parse
                s = cand
                break
    if s is None:
        s = pd.to_datetime(df[tcol], errors="coerce")
    out = (df.assign(**{tcol: s})
             .dropna(subset=[tcol])
             .sort_values(tcol)
             .set_index(tcol))
    out.index = out.index.tz_localize(None)
    out.index.name = "Time"
    return out


# ---------- readers ----------

def read_empatica_sensor(p: ParticipantPaths, sensor: Sensor) -> pd.DataFrame:
    """
    Reads your new headered CSVs:
      ACC:  datetime, acc_x, acc_y, acc_z  -> returns Var = sqrt(x^2+y^2+z^2)
      HR:   datetime, hr                   -> Var
      TEMP: datetime, temp                 -> Var
      EDA:  datetime, eda                  -> Var
      BVP:  datetime, bvp                  -> Var
    """
    path = p.sensor_path(sensor)
    df = _pd_read_csv_fast(path)

    # normalize header names to lowercase to find expected value column
    lower = {c.lower(): c for c in df.columns}

    if sensor == "ACC":
        # expect acc_x, acc_y, acc_z
        x = lower.get("acc_x")
        y = lower.get("acc_y")
        z = lower.get("acc_z")
        if not all([x, y, z]):
            raise ValueError(f"ACC file {path} must have columns: datetime, acc_x, acc_y, acc_z")
        # ensure numeric
        for c in (x, y, z):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = _to_time_index_smart(df, prefer_formats=["%Y-%m-%d %H:%M:%S.%f","%Y-%m-%d %H:%M:%S"])
        df["Var"] = np.sqrt(df[x]**2 + df[y]**2 + df[z]**2)
        return df[["Var"]]

    # single-channel sensors
    val_map = {
        "EDA": "eda",
        "TEMP": "temp",
        "HR": "hr",
        "BVP": "bvp",
    }
    if sensor in val_map:
        vname = val_map[sensor]
        vcol = lower.get(vname)
        if vcol is None:
            # fallback: try the second column if present
            vcol = df.columns[1] if df.shape[1] >= 2 else None
        if vcol is None:
            raise ValueError(f"{sensor} file {path} missing value column (expected '{vname}')")
        df[vcol] = pd.to_numeric(df[vcol], errors="coerce")
        df = _to_time_index_smart(df, prefer_formats=["%Y-%m-%d %H:%M:%S.%f","%Y-%m-%d %H:%M:%S"])
        return df[[vcol]].rename(columns={vcol: "Var"})

    raise ValueError(f"Unsupported sensor for read_empatica_sensor: {sensor}")

def read_dexcom(p: ParticipantPaths) -> pd.DataFrame:
    path = p.sensor_path("Dexcom")
    df = _pd_read_csv_fast(path)

    # Keep only glucose events (EGV)
    if "Event Type" in df.columns:
        df = df[df["Event Type"].astype(str).str.upper() == "EGV"]

    # Columns per Dexcom export
    tcol = "Timestamp (YYYY-MM-DDThh:mm:ss)" if "Timestamp (YYYY-MM-DDThh:mm:ss)" in df.columns else None
    gcol = "Glucose Value (mg/dL)" if "Glucose Value (mg/dL)" in df.columns else None

    # Fallbacks
    if tcol is None:
        for c in ("Timestamp", "Time", "Datetime"):
            if c in df.columns:
                tcol = c; break
    if gcol is None:
        for c in ("Glucose (mg/dL)", "Glucose"):
            if c in df.columns:
                gcol = c; break
    if tcol is None or gcol is None:
        tcol, gcol = df.columns[:2]

    out = df[[tcol, gcol]].rename(columns={tcol: "Time", gcol: "Glucose"})
    out["Time"] = pd.to_datetime(out["Time"], errors="coerce")
    out = out.dropna(subset=["Time"]).sort_values("Time").set_index("Time")
    return out[["Glucose"]]

def read_ibi(p: ParticipantPaths) -> pd.DataFrame:
    """
    IBI file: datetime, ibi  (ibi in seconds; irregular sampling)
    """
    path = p.sensor_path("IBI")
    df = _pd_read_csv_fast(path)  # already strips header whitespace

    # column lookup
    nm = {re.sub(r'[^a-z0-9]+', '', c.strip().lower()): c for c in df.columns}
    icol = nm.get("ibi") or (df.columns[1] if df.shape[1] >= 2 else None)
    if icol is None:
        raise ValueError(f"IBI file {path} missing 'ibi' column")

    # numeric values
    df[icol] = pd.to_numeric(df[icol], errors="coerce")

    # set a clean datetime index 
    df = _to_time_index_smart(
        df,
        prefer_formats=["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"]
    )

    return df[[icol]].rename(columns={icol: "IBI"})


