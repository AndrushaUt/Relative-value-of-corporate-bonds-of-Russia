"""Train OLS and Ridge baselines with the same walk-forward protocol as CatBoost.

Output:
    data/predictions_ols.csv    — date, isin, g_spread_actual, g_spread_predicted, residual
    data/predictions_ridge.csv  — same format

Methodology:
    * Same features as CatBoost no-lag (FEATURES_ALL, 38 features).
    * Sector one-hot encoded (CatBoost handles native, sklearn doesn't).
    * Numeric features median-imputed + standardized.
    * Walk-forward CV: 12-month train, 1-month step. 62 folds for 2021-01..2026-03.
    * OLS: ordinary least squares.
    * Ridge: RidgeCV with alpha in [0.1, 1.0, 10.0, 100.0] (inner CV).

Usage:
    python scripts/train_baselines.py            # train both
    python scripts/train_baselines.py --quick    # only first 12 folds (smoke test)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PANEL_PATH = DATA_DIR / "panel_engineered.csv"

# Те же фичи, что в CatBoost no-lag (см. scripts/optuna_search.py)
FEATURES_TIER1 = [
    "duration", "rating_numeric", "net_debt_ebitda",
    "key_rate", "ofz_slope", "rvi",
    "oil_brent", "usdrub", "sector", "issue_volume_rub",
]
FEATURES_TIER2 = [
    "imoex_return", "volume_ma20", "coupon_rate", "age_days",
    "imoex_realized_vol", "roe", "roa", "debt_to_equity",
    "rsbu_net_debt", "rsbu_ebit", "rsbu_icr",
    "is_state_owned", "time_to_maturity", "coupon_frequency",
    "ebitda_margin", "ebitda_yoy", "revenue_yoy",
    "rsbu_interest_paid", "rsbu_total_assets", "rsbu_revenue",
]
FEATURES_ENGINEERED = [
    "oil_brent_delta_30d", "usdrub_delta_30d", "key_rate_delta_30d",
    "imoex_return_ma5",
    "log_issue_volume", "log_age_days",
    "rating_x_duration", "leverage_x_rate",
]
FEATURES_ALL = FEATURES_TIER1 + FEATURES_TIER2 + FEATURES_ENGINEERED
CAT_FEATURES = ["sector"]
NUMERIC_FEATURES = [f for f in FEATURES_ALL if f not in CAT_FEATURES]
TARGET = "g_spread"


def walk_forward_split(df: pd.DataFrame, train_months: int = 12, step_months: int = 1):
    """Same as scripts/optuna_search.py."""
    dates = sorted(df["date"].dt.to_period("M").unique())
    for i in range(train_months, len(dates), step_months):
        train_periods = dates[max(0, i - train_months):i]
        test_period = dates[i]
        train_mask = df["date"].dt.to_period("M").isin(train_periods)
        test_mask = df["date"].dt.to_period("M") == test_period
        yield df[train_mask], df[test_mask]


def build_pipeline(model_name: str) -> Pipeline:
    """Build preprocessing + model pipeline.

    Args:
        model_name: 'ols' or 'ridge'.
    """
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CAT_FEATURES),
    ], remainder="drop")

    if model_name == "ols":
        regressor = LinearRegression()
    elif model_name == "ridge":
        regressor = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    else:
        raise ValueError(f"Unknown model {model_name}")

    return Pipeline([("prep", preprocessor), ("reg", regressor)])


def train_baseline(
    df: pd.DataFrame,
    model_name: str,
    quick: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Walk-forward training. Returns (predictions DataFrame, stats dict)."""
    print(f"\n=== Training {model_name.upper()} ===")
    folds = list(walk_forward_split(df))
    if quick:
        folds = folds[:12]
    print(f"Total folds: {len(folds)}")

    all_preds = []
    fold_metrics = []
    chosen_alphas = []
    t0 = time.time()

    for i, (train_df, test_df) in enumerate(folds, start=1):
        train_df = train_df.dropna(subset=[TARGET]).copy()
        test_df = test_df.dropna(subset=[TARGET]).copy()
        if len(train_df) < 100 or len(test_df) < 10:
            continue

        X_tr = train_df[FEATURES_ALL]
        y_tr = train_df[TARGET]
        X_te = test_df[FEATURES_ALL]
        y_te = test_df[TARGET]

        pipe = build_pipeline(model_name)
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        if model_name == "ridge":
            chosen_alphas.append(float(pipe.named_steps["reg"].alpha_))

        preds = test_df[["date", "isin"]].copy()
        preds["g_spread_actual"] = y_te.to_numpy()
        preds["g_spread_predicted"] = y_pred
        preds["residual"] = preds["g_spread_actual"] - preds["g_spread_predicted"]
        all_preds.append(preds)

        mae = mean_absolute_error(y_te, y_pred)
        r2 = r2_score(y_te, y_pred)
        fold_metrics.append({"fold": i, "n_test": len(test_df), "MAE": mae, "R2": r2})

        if i % 10 == 0 or i == len(folds):
            elapsed = time.time() - t0
            print(f"  fold {i:3d}/{len(folds)}: MAE={mae:.3f}, R²={r2:.3f}, "
                  f"elapsed {elapsed:.0f}s")

    preds_df = pd.concat(all_preds, ignore_index=True)
    fold_df = pd.DataFrame(fold_metrics)

    overall_mae = mean_absolute_error(preds_df["g_spread_actual"], preds_df["g_spread_predicted"])
    overall_r2 = r2_score(preds_df["g_spread_actual"], preds_df["g_spread_predicted"])
    stats = {
        "model": model_name,
        "n_folds": len(fold_df),
        "n_predictions": len(preds_df),
        "MAE_overall": overall_mae,
        "R2_overall": overall_r2,
        "MAE_mean_fold": fold_df["MAE"].mean(),
        "R2_mean_fold": fold_df["R2"].mean(),
        "elapsed_sec": time.time() - t0,
    }
    if chosen_alphas:
        stats["alphas_mean"] = float(np.mean(chosen_alphas))
        stats["alphas_median"] = float(np.median(chosen_alphas))

    print(f"\n{model_name.upper()} summary:")
    print(f"  Overall MAE: {overall_mae:.3f}")
    print(f"  Overall R²:  {overall_r2:.3f}")
    if chosen_alphas:
        print(f"  Ridge α median: {stats['alphas_median']}")

    return preds_df, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Only 12 folds (smoke test)")
    parser.add_argument("--models", default="ols,ridge",
                        help="Comma-separated: ols,ridge")
    args = parser.parse_args()

    print(f"Loading {PANEL_PATH}...")
    df = pd.read_csv(PANEL_PATH, parse_dates=["date"])
    df["sector"] = df["sector"].fillna("Unknown").astype(str)
    df.loc[df["sector"].str.strip() == "", "sector"] = "Unknown"
    print(f"  Shape: {df.shape}, bonds: {df['isin'].nunique()}")
    print(f"  Date range: {df['date'].min().date()} .. {df['date'].max().date()}")

    models = [m.strip() for m in args.models.split(",")]
    stats_all = []
    for m in models:
        preds_df, stats = train_baseline(df, m, quick=args.quick)
        out_path = DATA_DIR / f"predictions_{m}.csv"
        preds_df.to_csv(out_path, index=False)
        print(f"  Saved -> {out_path} ({len(preds_df):,} rows)")
        stats_all.append(stats)

    pd.DataFrame(stats_all).to_csv(DATA_DIR / "baseline_models_metrics.csv", index=False)
    print(f"\nSaved metrics -> {DATA_DIR / 'baseline_models_metrics.csv'}")

    print("\n=== Summary ===")
    print(pd.DataFrame(stats_all).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
