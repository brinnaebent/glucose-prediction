# src/glucose_fe/config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import yaml

@dataclass(frozen=True)
class Paths:
    root: Path
    medx_dir: str
    food_logs_dir: str
    out_dir: Path
    demographics_csv: Optional[str] = None  # relative to root

    @property
    def medx_root(self) -> Path:
        return self.root / self.medx_dir

    @property
    def food_root(self) -> Path:
        return self.root / self.food_logs_dir

    @property
    def demographics_path(self) -> Optional[Path]:
        return (self.root / self.demographics_csv) if self.demographics_csv else None

@dataclass(frozen=True)
class Windows:
    base_freq: str = "5min"
    base_feature_window: str = "10min"
    two_hr: str = "2h"
    eight_hr: str = "8h"
    day_24h: str = "24h"

@dataclass(frozen=True)
class PipelineConfig:
    paths: Paths
    windows: Windows
    participants: List[str]
    lookups: Dict[str, Dict[str, float | int]]
    engine: str = "pandas"               # "pandas" or "polars"
    use_bottleneck: bool = True
    pandas_csv_engine: Optional[str] = None
    pandas_dtype_backend: Optional[str] = None
    polars_threads: Optional[int] = None

    @staticmethod
    def from_yaml(path: str | Path) -> "PipelineConfig":
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        paths = Paths(
            root=Path(cfg["paths"]["root"]).expanduser(),
            medx_dir=cfg["paths"].get("medx_dir", "."),
            food_logs_dir=cfg["paths"].get("food_logs_dir", "."),
            out_dir=Path(cfg["paths"]["out_dir"]).expanduser(),
            demographics_csv=cfg["paths"].get("demographics_csv"),
        )

        pc = cfg.get("pipeline", {})
        wins = Windows(
            base_freq=pc.get("base_freq", "5min"),
            base_feature_window=pc.get("base_feature_window", "10min"),
            two_hr=pc.get("windows", {}).get("two_hr", "2h"),
            eight_hr=pc.get("windows", {}).get("eight_hr", "8h"),
            day_24h=pc.get("windows", {}).get("day_24h", "24h"),
        )

        participants = list(cfg.get("participants", {}).get("ids", []))
        lookups = cfg.get("lookups", {})

        return PipelineConfig(
            paths=paths,
            windows=wins,
            participants=participants,
            lookups=lookups,
            engine=pc.get("engine", "pandas"),
            use_bottleneck=bool(pc.get("use_bottleneck", True)),
            pandas_csv_engine=pc.get("pandas", {}).get("csv_engine"),
            pandas_dtype_backend=pc.get("pandas", {}).get("dtype_backend"),
            polars_threads=pc.get("polars", {}).get("threads"),
        )
