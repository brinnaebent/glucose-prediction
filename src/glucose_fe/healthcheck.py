# src/glucose_fe/healthcheck.py
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import pandas as pd
try:
    import yaml  # optional; used only if --rules is provided
except Exception:
    yaml = None

# ---------- rules (static bounds) ----------

@dataclass(frozen=True)
class Bounds:
    min: float | None = None
    max: float | None = None

def _is_numeric_dtype(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)

def _defaults() -> list[dict]:
    return [
        {"pattern": r"^Glucose$",                        "min": 30,   "max": 500},

        {"pattern": r"^HR_Mean$",                        "min": 20,   "max": 230},
        {"pattern": r"^HR_Min$",                         "min": 20,   "max": 230},
        {"pattern": r"^HR_Max$",                         "min": 20,   "max": 230},
        {"pattern": r"^HR_Std$",                         "min": 0,    "max": 50},

        {"pattern": r"^EDA_(Mean|Min|Max|Q1G|Q3G)$",     "min": 0,    "max": 100},
        {"pattern": r"^EDA_Std$",                        "min": 0,    "max": 100},

        {"pattern": r"^TEMP_(Mean|Min|Max|Q1G|Q3G)$",    "min": 15,   "max": 45},
        {"pattern": r"^TEMP_Std$",                       "min": 0,    "max": 10},

        {"pattern": r"^ACC_(Mean|Min|Max|Q1G|Q3G|Std)$", "min": 0,    "max": None},

        {"pattern": r"^PeakEDA$",                        "min": 0,    "max": None},
        {"pattern": r"^PeakEDA2hr_(sum|mean)$",          "min": 0,    "max": None},
        {"pattern": r"^Activity_bouts$",                 "min": 0,    "max": 1},
        {"pattern": r"^Activity1hr$",                    "min": 0,    "max": 12},
        {"pattern": r"^Activity24$",                     "min": 0,    "max": 1},

        {"pattern": r"^(rmssd|sdnn)$",                   "min": 0,    "max": 400},
        {"pattern": r"^(meanHrv|medianHrv)$",            "min": 300,  "max": 2000},
        {"pattern": r"^minHrv$",                         "min": 250,  "max": 1500},
        {"pattern": r"^maxHrv$",                         "min": 400,  "max": 2500},
        {"pattern": r"^pnnx$",                           "min": 0,    "max": 100},
        {"pattern": r"^nnx$",                            "min": 0,    "max": None},

        {"pattern": r"^(calorie|total_carb|sugar|protein)$",     "min": 0, "max": 5000},
        {"pattern": r"^(calorie|total_carb|sugar|protein)2hr$",  "min": 0, "max": 20000},
        {"pattern": r"^(calorie|total_carb|sugar|protein)8hr$",  "min": 0, "max": 40000},
        {"pattern": r"^(calorie|total_carb|sugar|protein)24hr$", "min": 0, "max": 80000},

        {"pattern": r"^Eat$",                          "min": 0, "max": 1},
        {"pattern": r"^Eatcnt2hr$",                    "min": 0, "max": 24},
        {"pattern": r"^Eatcnt8hr$",                    "min": 0, "max": 96},
        {"pattern": r"^Eatcnt24hr$",                   "min": 0, "max": 288},
        {"pattern": r"^Eatmean(2hr|8hr|24hr)$",        "min": 0, "max": 1},

        {"pattern": r"^(Minfrommid|WakeTime)$",        "min": 0, "max": 1440},
        {"pattern": r"^Hourfrommid$",                  "min": 0, "max": 24},

        # Explicitly avoid bounding skew/kurtosis:
        {"pattern": r"_Skew$", "min": None, "max": None},
        {"pattern": r"_Kurt$", "min": None, "max": None},
    ]

def _load_rules(path: Path | None) -> list[dict]:
    if path is None:
        return _defaults()
    if yaml is None:
        raise RuntimeError("pyyaml not installed; `pip install pyyaml` or omit --rules.")
    with open(path, "r") as f:
        user = yaml.safe_load(f) or {}
    user_rules = user.get("rules", [])
    return _defaults() + user_rules  

def _compile_bounds(columns: list[str], rules: list[dict]) -> dict[str, Bounds]:
    compiled: dict[str, Bounds] = {}
    for rule in rules:
        pat = re.compile(rule["pattern"])
        rmin = rule.get("min", None)
        rmax = rule.get("max", None)
        for col in columns:
            if pat.search(col):
                compiled[col] = Bounds(min=rmin, max=rmax)
    return compiled

# Columns that should never go below 0 (for robust lower cap clamping)
_NONNEG_PAT = re.compile(
    r"(_Std$|^ACC_|^EDA_(Mean|Min|Q1G|Q3G|Std)$|^TEMP_(Mean|Min|Q1G|Q3G|Std)$|"
    r"^PeakEDA|^Activity|^Eat|^calorie|^total_carb|^sugar|^protein)"
)

def _is_nonnegative_feature(col: str) -> bool:
    return _NONNEG_PAT.search(col) is not None

# ---------- stats ----------

def _per_participant_stats(
    df_pid: pd.DataFrame,
    pid: str,
    bounds: dict[str, Bounds],
    zero_epsilon: float = 0.0,
) -> pd.DataFrame:
    rows = len(df_pid)
    out_rows = []
    for col in df_pid.columns:
        if col in ("ID",):
            continue
        s = df_pid[col]
        dtype = str(s.dtype)
        non_null = s.notna().sum()
        missing = rows - non_null
        missing_pct = missing / rows if rows else np.nan

        zeros = zero_pct = np.nan
        mean = std = q25 = med = q75 = min_ = max_ = np.nan
        oob_lo = oob_hi = oob_cnt = oob_pct = np.nan

        if _is_numeric_dtype(s):
            s_num = pd.to_numeric(s, errors="coerce")
            if zero_epsilon == 0.0:
                zeros = (s_num == 0).sum(skipna=True)
            else:
                zeros = (s_num.fillna(0).abs() <= zero_epsilon).sum()
            zero_pct = zeros / rows if rows else np.nan

            if non_null > 0:
                mean = s_num.mean()
                std  = s_num.std(ddof=1)
                q25  = s_num.quantile(0.25)
                med  = s_num.quantile(0.50)
                q75  = s_num.quantile(0.75)
                min_ = s_num.min()
                max_ = s_num.max()

            # out-of-bounds check (static)
            b = bounds.get(col)
            if b and non_null > 0:
                lo_mask = (s_num < b.min) if b.min is not None else pd.Series(False, index=s_num.index)
                hi_mask = (s_num > b.max) if b.max is not None else pd.Series(False, index=s_num.index)
                bad = (lo_mask | hi_mask)
                oob_cnt = bad.sum(skipna=True)
                oob_pct = oob_cnt / rows if rows else np.nan
                oob_lo  = int(lo_mask.sum(skipna=True)) if b.min is not None else 0
                oob_hi  = int(hi_mask.sum(skipna=True)) if b.max is not None else 0

        out_rows.append({
            "ID": pid,
            "column": col,
            "dtype": dtype,
            "rows": rows,
            "non_null": int(non_null),
            "missing": int(missing),
            "missing_pct": float(missing_pct) if rows else np.nan,
            "zeros": int(zeros) if zeros == zeros and not np.isnan(zeros) else np.nan,
            "zero_pct": float(zero_pct) if zero_pct == zero_pct else np.nan,
            "mean": mean, "std": std, "min": min_, "q25": q25, "median": med, "q75": q75, "max": max_,
            "n_unique": int(df_pid[col].nunique(dropna=True)),
            "all_missing": bool(non_null == 0),
            "all_zero": bool(_is_numeric_dtype(s) and zeros == rows),
            "oob_count": int(oob_cnt) if oob_cnt == oob_cnt else np.nan,
            "oob_pct": float(oob_pct) if oob_pct == oob_pct else np.nan,
            "oob_low": int(oob_lo) if oob_lo == oob_lo else np.nan,
            "oob_high": int(oob_hi) if oob_hi == oob_hi else np.nan,
        })
    return pd.DataFrame(out_rows)

# ---------- robust row-level anomalies ----------

def _robust_anomalies_for_id(
    g: pd.DataFrame,
    pid: str,
    bounds: dict[str, Bounds],
    iqr_k: float,
    min_n: int,
    include_pat: re.Pattern | None = None,
    exclude_pat: re.Pattern | None = None,
) -> list[dict]:
    records: list[dict] = []
    idx_name = g.index.name or "Time"

    for col in g.columns:
        if col == "ID" or not _is_numeric_dtype(g[col]):
            continue
        if include_pat and not include_pat.search(col):
            continue
        if exclude_pat and exclude_pat.search(col):
            continue
        if col == "ID" or not _is_numeric_dtype(g[col]):
            continue
        s = pd.to_numeric(g[col], errors="coerce")
        s_valid = s.dropna()
        if len(s_valid) < max(min_n, 8):
            continue

        # Static bounds first
        b = bounds.get(col)
        if b is not None:
            if b.min is not None:
                bad = s < b.min
                for t, v in s[bad].items():
                    records.append({"ID": pid, "column": col, idx_name: t, "value": v,
                                    "reason": "static_low", "lower": b.min, "upper": b.max})
            if b.max is not None:
                bad = s > b.max
                for t, v in s[bad].items():
                    records.append({"ID": pid, "column": col, idx_name: t, "value": v,
                                    "reason": "static_high", "lower": b.min, "upper": b.max})

        # IQR bounds (per participant, per column)
        q1, q3 = s_valid.quantile([0.25, 0.75])
        iqr = (q3 - q1)
        if not np.isfinite(iqr) or iqr == 0:
            continue
        lo = q1 - iqr_k * iqr
        hi = q3 + iqr_k * iqr
        if _is_nonnegative_feature(col):
            lo = max(lo, 0.0)

        bad_lo = s < lo
        bad_hi = s > hi
        for t, v in s[bad_lo].items():
            records.append({"ID": pid, "column": col, idx_name: t, "value": v,
                            "reason": "robust_low", "lower": lo, "upper": hi})
        for t, v in s[bad_hi].items():
            records.append({"ID": pid, "column": col, idx_name: t, "value": v,
                            "reason": "robust_high", "lower": lo, "upper": hi})
    return records

# ---------- runner ----------

def run(
    out_dir: Path,
    parquet_file: str = "ALL_features_cleaned.parquet",
    missing_thresh: float = 0.2,
    zero_thresh: float = 0.8,
    zero_epsilon: float = 0.0,
    rules_yaml: Path | None = None,
    robust: bool = False,
    robust_iqr_k: float = 5.0,
    robust_min_n: int = 48,
    robust_include: str | None = None,
    robust_exclude: str | None = r"_Skew$|_Kurt$",
) -> tuple[Path, Path | None]:
    """
    Generate per-participant stats + optional robust row-level anomalies.

    robust_include: regex string; only columns matching are considered by robust detector (None = no filter)
    robust_exclude: regex string; columns matching are excluded by robust detector
                     (default excludes *_Skew and *_Kurt)
    """
    out_dir = out_dir.expanduser().resolve()
    report_dir = out_dir / "qa"
    report_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / parquet_file
    df = pd.read_parquet(path, engine="pyarrow")

    # Ensure datetime index if present
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        df = df.set_index("Time").sort_index()

    if "ID" not in df.columns:
        raise ValueError("Expected an 'ID' column in ALL_features_cleaned.parquet")

    # Compile bounds from defaults + optional YAML
    rules = _load_rules(rules_yaml)
    bounds = _compile_bounds([c for c in df.columns if c != "ID"], rules)

    # Compile robust include/exclude regex (default: exclude skew/kurt)
    inc_pat = re.compile(robust_include) if robust_include else None
    exc_pat = re.compile(robust_exclude) if robust_exclude else re.compile(r"_Skew$|_Kurt$")

    # Stats per participant (+ optional robust anomalies)
    all_stats: list[pd.DataFrame] = []
    row_anoms: list[dict] = []
    for pid, g in df.groupby("ID", sort=True):
        stats = _per_participant_stats(g, str(pid), bounds=bounds, zero_epsilon=zero_epsilon)
        all_stats.append(stats)
        stats.to_csv(report_dir / f"healthcheck_{pid}.csv", index=False)

        if robust:
            row_anoms.extend(
                _robust_anomalies_for_id(
                    g, str(pid), bounds=bounds, iqr_k=robust_iqr_k, min_n=robust_min_n,
                    include_pat=inc_pat, exclude_pat=exc_pat
                )
            )

    summary = pd.concat(all_stats, ignore_index=True)

    # High-level flags
    summary["flag_missing"] = summary["missing_pct"] >= float(missing_thresh)
    summary["flag_zeros"]   = summary["zero_pct"]    >= float(zero_thresh)
    summary["flag_oob"]     = (summary["oob_count"].fillna(0) > 0)

    combined_path = report_dir / "healthcheck_ALL.csv"
    summary.to_csv(combined_path, index=False)

    anomalies_path: Path | None = None
    if robust:
        anomalies = pd.DataFrame(row_anoms)
        if not anomalies.empty:
            # sort by ID then time if present
            tname = df.index.name or "Time"
            sort_cols = ["ID"] + ([tname] if tname in anomalies.columns else [])
            anomalies = anomalies.sort_values(sort_cols)
        anomalies_path = report_dir / "anomalies_rows.csv"
        anomalies.to_csv(anomalies_path, index=False)

    # Console summary
    print(f"\nWrote per-participant reports to: {report_dir}")
    print(f"Combined report: {combined_path}")
    if robust:
        print(f"Row-level anomalies: {anomalies_path}")

    print("\nWorst missingness (per ID, top 5):")
    for pid, g in summary.groupby("ID"):
        worst = g.sort_values("missing_pct", ascending=False).head(5)
        print(f"\nID {pid}")
        print(worst[["column", "missing_pct", "non_null", "rows", "dtype"]])

    print("\nColumns with any out-of-bounds values (by count):")
    oob_cols = (summary[summary["flag_oob"]]
                .groupby("column")["oob_count"].sum()
                .sort_values(ascending=False))
    print(oob_cols.head(20))

    if robust and anomalies_path and anomalies_path.exists():
        print("\nTop robust anomalies (by column):")
        an = pd.read_csv(anomalies_path, nrows=100000)
        if "column" in an.columns:
            print(an.groupby("column").size().sort_values(ascending=False).head(15))

    return combined_path, anomalies_path


def main():
    ap = argparse.ArgumentParser(description="Health-check stats for ALL_features_cleaned.parquet")
    ap.add_argument("--out", required=True, help="Path to output directory containing ALL_features_cleaned.parquet")
    ap.add_argument("--file", default="ALL_features_cleaned.parquet", help="Parquet file name")
    ap.add_argument("--missing-thresh", type=float, default=0.20, help="Flag columns with >= this fraction missing")
    ap.add_argument("--zero-thresh", type=float, default=0.80, help="Flag numeric columns with >= this fraction zeros")
    ap.add_argument("--zero-epsilon", type=float, default=0.0, help="Treat |x| <= epsilon as zero")
    ap.add_argument("--rules", type=str, default=None, help="YAML file with custom bounds")
    ap.add_argument("--robust", action="store_true", help="Enable robust per-participant IQR-based outlier flags")
    ap.add_argument("--iqr-k", type=float, default=5.0, help="IQR multiplier for robust bounds (default 5.0)")
    ap.add_argument("--robust-min-n", type=int, default=48, help="Min non-null points per (ID,col) to compute robust bounds")
    ap.add_argument("--robust-include", type=str, default=None, help="Regex: only columns matching are considered by robust detector")
    ap.add_argument("--robust-exclude", type=str, default=r"_Skew$|_Kurt$", help="Regex: columns matching are excluded by robust detector (default: exclude skew/kurtosis)")

    args = ap.parse_args()

    run(Path(args.out),
        parquet_file=args.file,
        missing_thresh=args.missing_thresh,
        zero_thresh=args.zero_thresh,
        zero_epsilon=args.zero_epsilon,
        rules_yaml=Path(args.rules) if args.rules else None,
        robust=args.robust,
        robust_iqr_k=args.iqr_k,
        robust_min_n=args.robust_min_n)

if __name__ == "__main__":
    main()





# Run: python -m src.glucose_fe.healthcheck --out ./out --robust --iqr-k 7 --robust-exclude "_Skew$|_Kurt$"

