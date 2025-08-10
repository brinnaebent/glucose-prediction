from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import PipelineConfig
from .features import FeatureWindows
from .io import ParticipantPaths, read_empatica_sensor, read_dexcom, read_ibi
from .features import compute_sensor_feature_frames, merge_on_index
from .glucose import resample_glucose_5min
from .hrv import hrv_window_features
from .stress import eda_peak_counts
from .wake import wake_features
from .food import food_features

log = logging.getLogger(__name__)


def _ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _pd_to_pl_time_var(df: pd.DataFrame) -> "pl.DataFrame":
    import polars as pl
    d = df.copy()
    d = d.reset_index()
    if d.columns[0] != "Time":
        d = d.rename(columns={d.columns[0]: "Time"})
    return pl.DataFrame(d[["Time", "Var"]]).with_columns(pl.col("Time").cast(pl.Datetime))

def _gender_to_binary(val) -> float:
    """Map gender to Female=0, Male=1. Handles text or 1/2 style codes."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().upper()
    if s in {"F", "FEMALE"}:
        return 0.0
    if s in {"M", "MALE"}:
        return 1.0
    return np.nan

def _pid_to_int(pid: str) -> int:
    """Convert '001' or '19-001' to an integer ID (1)."""
    digits = re.findall(r"\d+", str(pid))
    return int(digits[-1]) if digits else None

def process_participant(pid: str, cfg: PipelineConfig) -> Path:
    ppaths = ParticipantPaths(base=cfg.paths.medx_root, participant_id=pid)
    _ensure_out_dir(cfg.paths.out_dir)

    # 1) Read sensors
    eda = read_empatica_sensor(ppaths, "EDA")  # Var
    temp = read_empatica_sensor(ppaths, "TEMP")
    hr = read_empatica_sensor(ppaths, "HR")
    acc = read_empatica_sensor(ppaths, "ACC")
    dex = read_dexcom(ppaths)

    # 2) Glucose is already 5-min; normalize to exact 5-min grid
    g5 = resample_glucose_5min(dex)

    # 3) Rolling features per sensor (engine switch)
    fw = FeatureWindows(base_freq=cfg.windows.base_freq, base_feature_window=cfg.windows.base_feature_window)

    engine = (cfg.engine or "pandas").lower()
    if engine == "polars":
        import polars as pl
        from .features_polars import compute_sensor_feature_frames_polars, merge_on_time_polars

        sensors_pl = {
            "EDA": _pd_to_pl_time_var(eda),
            "TEMP": _pd_to_pl_time_var(temp),
            "HR": _pd_to_pl_time_var(hr),
            "ACC": _pd_to_pl_time_var(acc),
        }
        sensor_feats_pl = compute_sensor_feature_frames_polars(sensors_pl)
        base_times = pl.DataFrame({"Time": list(g5.index)})
        base_times = base_times.with_columns(pl.col("Time").cast(pl.Datetime))
        merged_pl = merge_on_time_polars(sensor_feats_pl, base_times)
        merged = merged_pl.to_pandas()
        merged = merged.set_index("Time")
        merged = merged.join(g5, how="left")
    else:
        sensor_feats = compute_sensor_feature_frames({
            "EDA": eda,
            "TEMP": temp,
            "HR": hr,
            "ACC": acc,
        }, fw=fw)
        merged = merge_on_index(sensor_feats | {"Glucose": g5}, base_freq=cfg.windows.base_freq)

    # 4) Stress (EDA peaks) and HRV features
    stress = eda_peak_counts(eda, fs_hz=4.0, base_freq=cfg.windows.base_freq)
    try:
        ibi = read_ibi(ppaths)
        hrv = hrv_window_features(ibi, freq=cfg.windows.base_freq, x_ms=50.0)
    except FileNotFoundError:
        hrv = pd.DataFrame(index=merged.index)

    features = merged.join(stress).join(hrv)

    # 5) Exercise/activity bouts — vectorized expanding means as in legacy logic
    if {"ACC_Mean", "HR_Mean"}.issubset(features.columns):
        acc_gt_past = features["ACC_Mean"] > features["ACC_Mean"].expanding(min_periods=2).mean().shift(1)
        hr_gt_past = features["HR_Mean"] > features["HR_Mean"].expanding(min_periods=2).mean().shift(1)
        bouts = (acc_gt_past & hr_gt_past).astype(int)
        features["Activity_bouts"] = bouts
        features["Activity24"] = bouts.rolling("24h", min_periods=1).mean()
        features["Activity1hr"] = bouts.rolling("1h", min_periods=1).sum()

    # 6) ACC rollups to match legacy names
    if {"ACC_Mean", "ACC_Max"}.issubset(features.columns):
        features["ACC_Mean_2hrs"] = features["ACC_Mean"].rolling("2h", min_periods=1).mean()
        features["ACC_Max_2hrs"] = features["ACC_Max"].rolling("2h", min_periods=1).max()

    # 7) Wake/sleep heuristics (adds SleepWake1hr/3hr/4hr and WakeTime)
    needed = {"ACC_Mean", "ACC_Std", "HR_Mean", "HR_Std"}
    if needed.issubset(features.columns):
        wk_in = features[list(needed)].copy()

        # ensure DatetimeIndex
        if not pd.api.types.is_datetime64_any_dtype(wk_in.index):
            wk_in.index = pd.to_datetime(wk_in.index, errors="coerce")

        # make the column wake_features expects
        wk_in["Date"] = wk_in.index.normalize()

        wk = wake_features(wk_in)
        features = features.join(wk[[c for c in wk.columns if c not in features.columns]])


    # 8) Food log features aligned to base grid
    food_path = ppaths.food_log_path()
    foodf = food_features(food_path, base_index=features.index)
    if not foodf.empty:
        features = features.join(foodf)

    # 9) Add demographics if available (fallback to lookups otherwise)
    gender_val = np.nan
    hba1c_val = np.nan

    if cfg.paths.demographics_path and cfg.paths.demographics_path.exists():
        demo = pd.read_csv(cfg.paths.demographics_path)
        # normalize IDs to 3-digit strings for matching (supports 001 / 1)
        demo["ID"] = demo["ID"].astype(str).str.zfill(3)
        row = demo.loc[demo["ID"] == pid]
        if not row.empty:
            r = row.iloc[0]
            if "Gender" in r.index:
                gender_val = _gender_to_binary(r["Gender"])
            if "HbA1c" in r.index:
                try:
                    hba1c_val = float(r["HbA1c"])
                except Exception:
                    hba1c_val = pd.to_numeric(r["HbA1c"], errors="coerce")
    else:
        # Optional: pull from config lookups if provided
        if "Gender" in cfg.lookups:
            gender_val = _gender_to_binary(cfg.lookups["Gender"].get(pid, np.nan))
        if "HbA1c" in cfg.lookups:
            hba1c_val = pd.to_numeric(cfg.lookups["HbA1c"].get(pid, np.nan), errors="coerce")

    if not np.isnan(gender_val):
        features["Gender"] = float(gender_val)
    if not np.isnan(hba1c_val):
        features["HbA1c"] = float(hba1c_val)


    # 10) Time-of-day features for parity
    idx = features.index
    features["Minfrommid"] = idx.hour * 60 + idx.minute
    features["Hourfrommid"] = idx.hour + idx.minute / 60.0

    # 11) Finalize
    features.index.name = "Time"

    # Keep string ID for grouping/LOPOCV splits
    features["ID"] = pid

    # Add numeric ID feature for modeling
    id_num = _pid_to_int(pid)
    if id_num is not None:
        features["ID_num"] = int(id_num)

    # Time-of-day already added above
    out_path = cfg.paths.out_dir / f"{pid}_features.parquet"
    features.to_parquet(out_path, engine="pyarrow", index=True)
    log.info("Wrote %s with %d rows and %d columns", out_path, len(features), features.shape[1])
    return out_path



def compile_all(cfg: PipelineConfig) -> Path:
    _ensure_out_dir(cfg.paths.out_dir)
    files = list(cfg.paths.out_dir.glob("*_features.parquet"))
    if not files:
        raise FileNotFoundError("No per-participant parquet files found to compile.")
    parts = [pd.read_parquet(p) for p in files]
    df = pd.concat(parts, axis=0, ignore_index=False).sort_index()
    compiled = df.loc[df["Glucose"].notna()]
    out_path = cfg.paths.out_dir / "ALL_features_cleaned.parquet"
    compiled.to_parquet(out_path, engine="pyarrow")
    return out_path

    out_path = cfg.paths.out_dir / "ALL_features_cleaned.parquet"
    compiled.to_parquet(out_path, engine="pyarrow")
    return out_path


def run_parallel(cfg: PipelineConfig, max_workers: int | None = None) -> list[Path]:
    _ensure_out_dir(cfg.paths.out_dir)
    results: list[Path] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(process_participant, pid, cfg): pid for pid in cfg.participants}
        for fut in as_completed(futs):
            pid = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                logging.exception("Participant %s failed: %s", pid, e)
    return results