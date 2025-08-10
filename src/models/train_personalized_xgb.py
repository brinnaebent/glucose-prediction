from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Dict
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from .config import XGBLOOCVConfig
from .utils import ensure_dir, setup_logging, rmse, mape

def _load_sorted(df_path: Path, id_col: str, time_col: str) -> pd.DataFrame:
    df = pd.read_parquet(df_path, engine="pyarrow")

    # If Time isn't a column, pull it from the index
    if time_col not in df.columns:
        # ensure the index has a name; if not, treat it as Time
        if df.index.name is None:
            df.index.name = time_col
        if df.index.name == time_col:
            df = df.reset_index()
        else:
            raise KeyError(f"'{time_col}' not found as column or index in {df_path.name}")

    # canonicalize types & sort
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df[id_col] = df[id_col].astype(str)
    return df.sort_values([id_col, time_col]).reset_index(drop=True)

def _median_impute_for_rf(X: pd.DataFrame) -> pd.DataFrame:
    med = X.median(numeric_only=True)
    return X.fillna(med)

def _rf_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int,
    random_state: int,
    n_jobs: int,
    threshold: float
) -> List[str]:
    X_imp = _median_impute_for_rf(X_train)
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        oob_score=False,
    )
    rf.fit(X_imp, y_train.to_numpy())
    imp = pd.Series(rf.feature_importances_, index=X_train.columns, name="importance")
    keep = imp[imp >= threshold].index.tolist()
    return keep, imp.sort_values(ascending=False)

def main():
    ap = argparse.ArgumentParser(description="EXACT methods: Personalized 50/50 XGBoost with per-participant RF feature selection.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = XGBLOOCVConfig.from_yaml(args.config)
    setup_logging()

    out = cfg.out_dir
    models_dir = out / "models"
    preds_dir  = out / "preds"
    fl_dir     = out / "feature_lists"
    imp_dir    = out / "feature_importances"
    for d in (out, models_dir, preds_dir, fl_dir, imp_dir):
        ensure_dir(d)

    df = _load_sorted(cfg.data_path, cfg.id_col, cfg.time_col)
    ids: List[str] = [str(x) for x in df[cfg.id_col].astype(str).unique()]

    rows, metrics_rows, all_importances = [], [], []

    for pid in ids:
        print(f"\n=== Personalized 50/50: {pid} ===")
        p = df[df[cfg.id_col].astype(str) == pid].copy()
        n = len(p)
        if n < 20:
            print(f"Skip {pid}: not enough rows ({n}).")
            continue
        split = n // 2
        train_p = p.iloc[:split].copy()
        test_p  = p.iloc[split:].copy()

        feature_cols = [c for c in df.columns if c not in set([cfg.id_col, cfg.target, *cfg.drop_cols])]

        # RF FS on first half (train) ONLY
        X_train_fs = train_p[feature_cols]
        y_train    = train_p[cfg.target].astype(np.float32)
        keep_cols, imp_series = _rf_feature_selection(
            X_train_fs, y_train,
            n_estimators=cfg.rf_n_estimators,
            random_state=cfg.rf_random_state,
            n_jobs=cfg.n_jobs,
            threshold=cfg.fs_threshold,
        )
        imp_df = imp_series.reset_index().rename(columns={"index": "feature"})
        imp_df.insert(0, "ID", pid)
        imp_df.to_csv(imp_dir / f"rf_importances_{pid}.csv", index=False)
        all_importances.append(imp_df)

        X_train = train_p[keep_cols].astype(np.float32).values
        X_test  = test_p[keep_cols].astype(np.float32).values
        y_test  = test_p[cfg.target].astype(np.float32).values

        model = XGBRegressor(**cfg.base_params)
        model.fit(X_train, y_train.values, verbose=False)
        model.save_model(models_dir / f"xgb_personalized_{pid}.json")

        y_pred = model.predict(X_test)
        r = rmse(y_test, y_pred)
        m = mape(y_test, y_pred)
        acc = 100.0 - m

        print(f"{pid}: RMSE={r:.3f}  MAPE={m:.2f}%  ACC={acc:.2f}%  (n_test={len(y_test)})")

        rows.append(pd.DataFrame({
            cfg.time_col: test_p[cfg.time_col].values,
            cfg.id_col:   test_p[cfg.id_col].astype(str).values,
            "y_true":     y_test,
            "y_pred":     y_pred.astype(np.float32),
            "fold_id":    pid
        }))

        with open(fl_dir / f"features_{pid}.json", "w") as f:
            json.dump(keep_cols, f, indent=2)

        metrics_rows.append({
            "ID": pid,
            "n_test": int(len(y_test)),
            "rmse": float(r),
            "mape": float(m),
            "accuracy": float(acc),
            "n_features": int(len(keep_cols)),
        })

    preds = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    metrics = pd.DataFrame(metrics_rows)
    preds.to_parquet(preds_dir / "preds_personalized.parquet", index=False)
    metrics.to_csv(out / "metrics_personalized.csv", index=False)

    if all_importances:
        agg = (pd.concat(all_importances, ignore_index=True)
                 .groupby("feature", as_index=False)["importance"].mean()
                 .sort_values("importance", ascending=False))
        agg.to_csv(out / "rf_importances_mean_across_participants.csv", index=False)

    if not metrics.empty:
        print("\n=== Personalized Summary ===")
        print(metrics[["ID","n_test","rmse","mape","accuracy","n_features"]])
        print(f"\nMean RMSE: {metrics['rmse'].mean():.3f} ± {metrics['rmse'].std():.3f}")
        print(f"Mean MAPE: {metrics['mape'].mean():.2f}% ± {metrics['mape'].std():.2f}%")
        print(f"Mean ACC : {metrics['accuracy'].mean():.2f}% ± {metrics['accuracy'].std():.2f}%")

if __name__ == "__main__":
    main()

#Run: python -m src.models.train_personalized_xgb --config configs/model_personalized.yaml