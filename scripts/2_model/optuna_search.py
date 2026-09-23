#!/usr/bin/env python3
"""
Optuna hyperparameter search for CatBoost G-spread model.

Standalone script — no Jupyter, no extra dependencies beyond catboost/optuna/pandas/numpy.
Designed to run on a powerful server (96 cores).

Usage:
    python optuna_search.py                    # 20 trials, 14 folds
    python optuna_search.py --n-trials 50      # more trials
    python optuna_search.py --all-folds        # use all optuna folds (slower)

Output: prints best params + saves to best_params.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor

# Feature lists (must match notebook 04)

FEATURES_TIER1 = [
    'duration', 'rating_numeric', 'net_debt_ebitda',
    'key_rate', 'ofz_slope', 'rvi',
    'oil_brent', 'usdrub', 'sector', 'issue_volume_rub',
]

FEATURES_TIER2 = [
    'imoex_return', 'volume_ma20', 'coupon_rate', 'age_days',
    'imoex_realized_vol', 'roe', 'roa', 'debt_to_equity',
    'rsbu_net_debt', 'rsbu_ebit', 'rsbu_icr',
    'is_state_owned', 'time_to_maturity', 'coupon_frequency',
    'ebitda_margin', 'ebitda_yoy', 'revenue_yoy',
    'rsbu_interest_paid', 'rsbu_total_assets', 'rsbu_revenue',
]

FEATURES_ENGINEERED = [
    'oil_brent_delta_30d', 'usdrub_delta_30d', 'key_rate_delta_30d',
    'imoex_return_ma5',
    # 'g_spread_lag1',
    'log_issue_volume', 'log_age_days',
    'rating_x_duration', 'leverage_x_rate',
]

FEATURES_ALL = FEATURES_TIER1 + FEATURES_TIER2 + FEATURES_ENGINEERED
CAT_FEATURES = ['sector']
TARGET = 'g_spread'


# Walk-forward split

def walk_forward_split(df, train_months=12, step_months=1):
    dates = sorted(df['date'].dt.to_period('M').unique())
    for i in range(train_months, len(dates), step_months):
        train_periods = dates[max(0, i - train_months):i]
        test_period = dates[i]
        train_mask = df['date'].dt.to_period('M').isin(train_periods)
        test_mask = df['date'].dt.to_period('M') == test_period
        yield df[train_mask], df[test_mask]


def main():
    parser = argparse.ArgumentParser(description="Optuna CatBoost hyperparameter search")
    parser.add_argument("--data", default="panel_engineered.csv", help="Path to panel CSV")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--all-folds", action="store_true", help="Use all folds (not every 3rd)")
    parser.add_argument("--output", default="best_params3.json", help="Output JSON file")
    args = parser.parse_args()

    print(f"Loading {args.data}...")
    df = pd.read_csv(args.data, parse_dates=['date'])
    df['sector'] = df['sector'].fillna('Unknown').astype(str)
    df.loc[df['sector'].str.strip() == '', 'sector'] = 'Unknown'
    print(f"  Shape: {df.shape}, bonds: {df['isin'].nunique()}")

    cat_feature_indices = [FEATURES_ALL.index(f) for f in CAT_FEATURES]
    print(f"  Features: {len(FEATURES_ALL)}, cat_indices: {cat_feature_indices}")

    # Collect folds (first 2/3 for Optuna)
    print("Collecting walk-forward folds...")
    all_folds = []
    for train_df, test_df in walk_forward_split(df):
        X_tr = train_df[FEATURES_ALL].copy()
        y_tr = train_df[TARGET]
        X_te = test_df[FEATURES_ALL].copy()
        y_te = test_df[TARGET]
        all_folds.append((X_tr, y_tr, X_te, y_te))

    n_optuna = int(len(all_folds) * 2 / 3)
    optuna_folds = all_folds[:n_optuna]

    if not args.all_folds:
        optuna_folds = optuna_folds[::3]

    print(f"  Total folds: {len(all_folds)}, Optuna folds: {len(optuna_folds)}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 300, 1500),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_seed': 42,
            'verbose': 0,
            'cat_features': cat_feature_indices,
            'allow_writing_files': False,
            'thread_count': -1,
        }

        maes = []
        for X_tr, y_tr, X_te, y_te in optuna_folds:
            model = CatBoostRegressor(**params)
            model.fit(X_tr, y_tr, verbose=0)
            y_pred = model.predict(X_te)
            maes.append(np.mean(np.abs(y_te - y_pred)))

        return np.mean(maes)

    print(f"\nStarting Optuna: {args.n_trials} trials, {len(optuna_folds)} folds each...")
    t0 = time.time()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} minutes")
    print(f"Best MAE: {study.best_value:.4f}")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    output = {
        'best_mae': study.best_value,
        'best_params': study.best_params,
        'n_trials': args.n_trials,
        'n_folds': len(optuna_folds),
        'elapsed_minutes': round(elapsed / 60, 1),
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
