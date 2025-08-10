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
    ap = argparse.ArgumentParser(description="EXACT methods: Population LOPOCV XGBoost (no warm start) with per-fold RF feature selection.")
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

    fold_rows: List[pd.DataFrame] = []
    fold_metrics: List[Dict] = []
    all_importances: List[pd.DataFrame] = []

    for pid in ids:
        print(f"\n=== LOPOCV (population): leave out {pid} ===")
        train_df = df[df[cfg.id_col].astype(str) != pid].copy()
        test_df  = df[df[cfg.id_col].astype(str) == pid].copy()

        # features
        feature_cols = [c for c in df.columns if c not in set([cfg.id_col, cfg.target, *cfg.drop_cols])]
        # RF feature selection on TRAIN-ONLY
        X_train_fs = train_df[feature_cols]
        y_train    = train_df[cfg.target].astype(np.float32)
        keep_cols, imp_series = _rf_feature_selection(
            X_train_fs, y_train,
            n_estimators=cfg.rf_n_estimators,
            random_state=cfg.rf_random_state,
            n_jobs=cfg.n_jobs,
            threshold=cfg.fs_threshold,
        )

        # save per-fold importances
        imp_df = imp_series.reset_index().rename(columns={"index": "feature"})
        imp_df.insert(0, "ID_fold", pid)
        imp_df.to_csv(imp_dir / f"rf_importances_{pid}.csv", index=False)
        all_importances.append(imp_df)

        # build matrices
        X_train = train_df[keep_cols].astype(np.float32).values
        X_test  = test_df[keep_cols].astype(np.float32).values
        y_test  = test_df[cfg.target].astype(np.float32).values

        # XGB with fixed hyperparams; no early stopping
        model = XGBRegressor(**cfg.base_params)
        model.fit(X_train, y_train.values, verbose=False)
        model.save_model(models_dir / f"xgb_population_{pid}.json")

        y_pred = model.predict(X_test)
        fold_rmse = rmse(y_test, y_pred)
        fold_mape = mape(y_test, y_pred)
        fold_acc  = 100.0 - fold_mape

        print(f"{pid}: RMSE={fold_rmse:.3f}  MAPE={fold_mape:.2f}%  ACC={fold_acc:.2f}%  (n_test={len(y_test)})")

        # collect predictions
        fold_out = pd.DataFrame({
            cfg.time_col: test_df[cfg.time_col].values,
            cfg.id_col:   test_df[cfg.id_col].astype(str).values,
            "y_true":     y_test,
            "y_pred":     y_pred.astype(np.float32),
            "fold_id":    pid
        })
        fold_rows.append(fold_out)

        # save feature list actually used
        with open(fl_dir / f"features_{pid}.json", "w") as f:
            json.dump(keep_cols, f, indent=2)

        fold_metrics.append({
            "ID": pid,
            "n_test": int(len(y_test)),
            "rmse": float(fold_rmse),
            "mape": float(fold_mape),
            "accuracy": float(fold_acc),
            "n_features": int(len(keep_cols)),
        })

    preds = pd.concat(fold_rows, ignore_index=True) if fold_rows else pd.DataFrame()
    metrics = pd.DataFrame(fold_metrics)
    preds.to_parquet(preds_dir / "preds_population_lopo.parquet", index=False)
    metrics.to_csv(out / "metrics_population_lopo.csv", index=False)

    # aggregate importances (mean across folds)
    if all_importances:
        agg = (pd.concat(all_importances, ignore_index=True)
                 .groupby("feature", as_index=False)["importance"].mean()
                 .sort_values("importance", ascending=False))
        agg.to_csv(out / "rf_importances_mean_across_folds.csv", index=False)

    if not metrics.empty:
        print("\n=== Population LOPOCV Summary ===")
        print(metrics[["ID","n_test","rmse","mape","accuracy","n_features"]])
        print(f"\nMean RMSE: {metrics['rmse'].mean():.3f} ± {metrics['rmse'].std():.3f}")
        print(f"Mean MAPE: {metrics['mape'].mean():.2f}% ± {metrics['mape'].std():.2f}%")
        print(f"Mean ACC : {metrics['accuracy'].mean():.2f}% ± {metrics['accuracy'].std():.2f}%")

if __name__ == "__main__":
    main()

# Run: python -m src.models.train_population_xgb --config configs/model_loocv.yaml
