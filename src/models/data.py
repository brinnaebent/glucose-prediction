from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from .config import XGBLOOCVConfig

def load_all_features(path: Path, id_col: str, time_col: str) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.sort_values([id_col, time_col]).reset_index(drop=True)
    return df

def _apply_feature_selection(
    X_cols: List[str],
    cfg: XGBLOOCVConfig,
    left_out_id: str
) -> List[str]:
    if not cfg.feature_importances_csv or cfg.feature_select_threshold is None:
        return X_cols

    imp = pd.read_csv(cfg.feature_importances_csv)
    # find id column
    idc = None
    for c in imp.columns:
        if c.lower() in ("id", "ids", "participant", "subject"):
            idc = c
            break
    if idc is None:
        raise ValueError("feature_importances_csv must include an id column (id/ids/participant/subject).")
    if "value" not in imp.columns or "importances" not in imp.columns:
        raise ValueError("feature_importances_csv must have columns: value, importances, and an id column.")

    imp_this = imp[imp[idc].astype(str) == str(left_out_id)]
    if imp_this.empty:
        return X_cols

    drop_set = set(imp_this.loc[imp_this["importances"] < cfg.feature_select_threshold, "value"].astype(str))
    return [c for c in X_cols if c not in drop_set]

def build_xy(
    df: pd.DataFrame,
    cfg: XGBLOOCVConfig,
    left_out_id: str
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    idc, tc, y = cfg.id_col, cfg.time_col, cfg.target

    train_df = df[df[idc].astype(str) != str(left_out_id)].copy()
    cv_df    = df[df[idc].astype(str) == str(left_out_id)].copy()

    feature_cols = [c for c in df.columns if c not in set([idc, y, *cfg.drop_cols])]
    feature_cols = _apply_feature_selection(feature_cols, cfg, left_out_id)

    # float32 cast for speed/memory; XGB handles NaN
    for c in feature_cols + [y]:
        if c in train_df.columns and pd.api.types.is_numeric_dtype(train_df[c]):
            train_df[c] = train_df[c].astype(np.float32)
        if c in cv_df.columns and pd.api.types.is_numeric_dtype(cv_df[c]):
            cv_df[c] = cv_df[c].astype(np.float32)

    return train_df[[*feature_cols, y, idc, tc]], cv_df[[*feature_cols, y, idc, tc]], feature_cols
