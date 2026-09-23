"""Retrain CatBoost no-lag with new features from Bryzgalova-Pelger-Zhu (NBER 2024).

New features:
    1. downside_variance_60d - variance of negative returns only (downside risk)
    2. return_skewness_60d  - skewness of total returns (asymmetric distribution)
    3. short_term_reversal  - past 30-day cumulative return (signed)

Same hyperparameters as catboost_no_lag.cbm (no Optuna re-tuning):
    depth=10, iterations=1005, learning_rate=0.0846, l2_leaf_reg=1.25,
    subsample=0.716, random_seed=42

Output:
    data/predictions_no_lag_v2.csv
    models/catboost_no_lag_v2.cbm
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PANEL_PATH = DATA_DIR / "panel_engineered.csv"

# Original features (from optuna_search.py)
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
# NEW Bryzgalova-Pelger-Zhu features
FEATURES_NEW = [
    "downside_variance_60d",
    "return_skewness_60d",
    "short_term_reversal_30d",
]
FEATURES_ALL = FEATURES_TIER1 + FEATURES_TIER2 + FEATURES_ENGINEERED + FEATURES_NEW
CAT_FEATURES = ["sector"]
TARGET = "g_spread"

# Hyperparameters from existing catboost_no_lag.cbm (no Optuna re-run)
HP = dict(
    depth=10, iterations=1005, learning_rate=0.0846, l2_leaf_reg=1.25,
    subsample=0.716, random_seed=42, verbose=0,
    allow_writing_files=False, thread_count=-1,
)


def compute_new_features(panel: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Compute downside variance, skewness, short-term reversal from trades."""
    trades = trades.sort_values(["isin", "date"]).copy()
    trades["log_p"] = np.log(trades["close_price"])
    trades["ret"] = trades.groupby("isin")["log_p"].diff()
    trades["ret"] = trades["ret"].clip(lower=-0.20, upper=0.20)
    trades["neg_ret_sq"] = np.where(trades["ret"] < 0, trades["ret"] ** 2, 0.0)

    def _rolling_stats(grp: pd.DataFrame) -> pd.DataFrame:
        r = grp["ret"]
        neg_sq = grp["neg_ret_sq"]
        dvar = neg_sq.rolling(60, min_periods=20).mean()
        skew = r.rolling(60, min_periods=20).skew()
        return pd.DataFrame({
            "downside_variance_60d": dvar.values,
            "return_skewness_60d": skew.values,
        }, index=grp.index)

    print("Computing downside variance + skewness per isin...")
    stats = trades.groupby("isin", group_keys=False).apply(_rolling_stats)
    trades = trades.join(stats)

    # short-term reversal: 30-day cumulative log return (signed)
    print("Computing short-term reversal (30d)...")
    trades["log_p_lag30"] = trades.groupby("isin")["log_p"].shift(30)
    trades["short_term_reversal_30d"] = (
        (trades["log_p"] - trades["log_p_lag30"]).clip(lower=-0.50, upper=0.50)
    )

    new_features = trades[
        ["date", "isin", "downside_variance_60d", "return_skewness_60d",
         "short_term_reversal_30d"]
    ].copy()

    out = panel.merge(new_features, on=["date", "isin"], how="left")
    print(f"  downside_variance: coverage {out['downside_variance_60d'].notna().mean():.1%}")
    print(f"  return_skewness:  coverage {out['return_skewness_60d'].notna().mean():.1%}")
    print(f"  short_term_rev:   coverage {out['short_term_reversal_30d'].notna().mean():.1%}")
    return out


def walk_forward_split(df: pd.DataFrame, train_months: int = 12, step_months: int = 1):
    dates = sorted(df["date"].dt.to_period("M").unique())
    for i in range(train_months, len(dates), step_months):
        train_periods = dates[max(0, i - train_months):i]
        test_period = dates[i]
        train_mask = df["date"].dt.to_period("M").isin(train_periods)
        test_mask = df["date"].dt.to_period("M") == test_period
        yield df[train_mask], df[test_mask]


def main() -> None:
    print("Loading panel + trades...")
    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])
    panel["sector"] = panel["sector"].fillna("Unknown").astype(str)
    panel.loc[panel["sector"].str.strip() == "", "sector"] = "Unknown"
    trades = pd.read_csv(
        DATA_DIR / "trades_daily.csv", parse_dates=["date"],
        usecols=["date", "isin", "close_price"],
    )
    print(f"  panel: {panel.shape}, trades: {trades.shape}")

    panel = compute_new_features(panel, trades)

    cat_idx = [FEATURES_ALL.index(f) for f in CAT_FEATURES]
    print(f"  features: {len(FEATURES_ALL)} (incl. {len(FEATURES_NEW)} new)")
    print(f"  cat indices: {cat_idx}")

    folds = list(walk_forward_split(panel))
    print(f"\nWalk-forward folds: {len(folds)}")
    print("=" * 70)

    all_preds = []
    fold_metrics = []
    t0 = time.time()

    for i, (train_df, test_df) in enumerate(folds, start=1):
        train_df = train_df.dropna(subset=[TARGET]).copy()
        test_df = test_df.dropna(subset=[TARGET]).copy()
        if len(train_df) < 100 or len(test_df) < 10:
            continue

        X_tr = train_df[FEATURES_ALL].copy()
        y_tr = train_df[TARGET]
        X_te = test_df[FEATURES_ALL].copy()
        y_te = test_df[TARGET]
        for c in CAT_FEATURES:
            X_tr[c] = X_tr[c].fillna("Unknown").astype(str)
            X_te[c] = X_te[c].fillna("Unknown").astype(str)

        model = CatBoostRegressor(cat_features=cat_idx, **HP)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        preds = test_df[["date", "isin"]].copy()
        preds["g_spread_actual"] = y_te.to_numpy()
        preds["g_spread_predicted"] = y_pred
        preds["residual"] = preds["g_spread_actual"] - preds["g_spread_predicted"]
        all_preds.append(preds)

        mae = mean_absolute_error(y_te, y_pred)
        r2 = r2_score(y_te, y_pred)
        fold_metrics.append({"fold": i, "n_test": len(test_df), "MAE": mae, "R2": r2})

        if i % 5 == 0 or i == len(folds):
            elapsed = time.time() - t0
            print(f"  fold {i:3d}/{len(folds)}: MAE={mae:.3f}, R²={r2:.3f}, "
                  f"elapsed {elapsed:.0f}s")

    preds_df = pd.concat(all_preds, ignore_index=True)
    fold_df = pd.DataFrame(fold_metrics)

    overall_mae = mean_absolute_error(preds_df["g_spread_actual"], preds_df["g_spread_predicted"])
    overall_r2 = r2_score(preds_df["g_spread_actual"], preds_df["g_spread_predicted"])

    print("\n" + "=" * 70)
    print("CatBoost v2 SUMMARY")
    print("=" * 70)
    print(f"  Overall MAE: {overall_mae:.3f}")
    print(f"  Overall R²:  {overall_r2:.3f}")
    print(f"  Mean fold MAE: {fold_df['MAE'].mean():.3f}")
    print(f"  Mean fold R²:  {fold_df['R2'].mean():.3f}")

    out_path = DATA_DIR / "predictions_no_lag_v2.csv"
    preds_df.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path} ({len(preds_df):,} rows)")

    fold_df.to_csv(DATA_DIR / "catboost_v2_folds.csv", index=False)

    # Save final model (re-fit on last available fold for inspection)
    print("  (Models per fold not saved; only predictions used downstream)")
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")

    print("\nFor comparison:")
    print("  no_lag (original): R²=0.524, MAE=3.91")


if __name__ == "__main__":
    main()
