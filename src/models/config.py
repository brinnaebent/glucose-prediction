from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Dict
import os, yaml

def _cpu_jobs_default() -> int:
    return max(1, (os.cpu_count() or 1) - 1)

@dataclass
class XGBLOOCVConfig:
    data_path: Path
    out_dir: Path

    id_col: str = "ID"
    time_col: str = "Time"
    target: str = "Glucose"
    drop_cols: Sequence[str] = field(default_factory=lambda: ["Time"])

    warm_start_frac: float = 0.0

    seed: int = 42
    n_jobs: int = field(default_factory=_cpu_jobs_default)

    # Per-fold RandomForest feature selection
    fs_threshold: float = 0.005
    rf_n_estimators: int = 1000
    rf_random_state: int = 42

    # XGBoost params (FROM  PAPER: depth=6, n_estimators=100, lr=0.1, subsample=1, colsample=1)
    base_params: Dict = field(default_factory=lambda: dict(
        max_depth=6,
        n_estimators=100,
        learning_rate=0.1,
        min_child_weight=1.0,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=_cpu_jobs_default(),
        random_state=42,
    ))
    
    base_early_stopping_rounds: Optional[int] = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "XGBLOOCVConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        for k in ("data_path", "out_dir"):
            if k in raw:
                raw[k] = Path(raw[k])
        cfg = cls(**raw)
        cfg.base_params["n_jobs"] = cfg.n_jobs
        return cfg

