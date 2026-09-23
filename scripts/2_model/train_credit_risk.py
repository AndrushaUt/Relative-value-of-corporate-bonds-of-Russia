"""Train credit risk classifier (CatBoost + MLP) — predict default/downgrade.

Methodology:
    Target = 1 if rating_numeric drops by ≥1 within next 180 calendar days
             OR price drops >15% within next 30 days (default proxy)
             else 0.

    Features = те же 38 фичей что у CatBoost regression (FEATURES_ALL).
    Walk-forward CV: 12 mo train / 1 mo test, ~62 folds.

    Two models:
        1. CatBoostClassifier (ML / GBM)
        2. PyTorch MLP 3-layer (DL)

    Pre-registered hyperparameters (no tuning on holdout):
        CatBoost: depth=6, iterations=500, learning_rate=0.05
        MLP: hidden=[64,32,16], dropout=0.2, lr=1e-3, 100 epochs, early stop

Output:
    data/credit_risk_predictions_cb.csv  — date, isin, default_proba
    data/credit_risk_predictions_mlp.csv — same
    data/credit_risk_metrics.csv         — AUC, precision@top-decile per fold
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PANEL_PATH = DATA_DIR / "panel_engineered.csv"
TRADES_PATH = DATA_DIR / "trades_daily.csv"

# Same features as regression (from optuna_search.py)
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

DOWNGRADE_WINDOW_DAYS = 180
PRICE_DROP_WINDOW_DAYS = 30
PRICE_DROP_THRESHOLD = -0.15  # > 15% drop


def build_target(panel: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Construct binary target: 1 = adverse credit event in next 180d.

    Two parts:
    a) rating downgrade (rating_numeric drop ≥1 within next 180d)
    b) price crash (close_price drop > 15% within next 30d) — proxy for default
    """
    panel = panel.sort_values(["isin", "date"]).copy()
    # Future rating after 180 days
    panel["date_plus_180"] = panel["date"] + pd.Timedelta(days=DOWNGRADE_WINDOW_DAYS)
    # Forward rolling min rating: для каждой даты — какой будет rating через 180д
    # Делаем merge: для каждой строки находим min rating среди записей этой ISIN с date in [d+1, d+180]
    rows = []
    for isin, grp in panel.groupby("isin"):
        grp = grp.sort_values("date").reset_index(drop=True)
        ratings = grp["rating_numeric"].to_numpy()
        dates = grp["date"].to_numpy()
        future_min = np.full(len(grp), np.nan)
        for i in range(len(grp)):
            if not np.isfinite(ratings[i]):
                continue
            end_date = dates[i] + np.timedelta64(DOWNGRADE_WINDOW_DAYS, 'D')
            # Найти min rating в окне (i+1, ..., где date <= end_date)
            future_idx = np.where((dates > dates[i]) & (dates <= end_date))[0]
            if len(future_idx) > 0:
                fut_r = ratings[future_idx]
                fut_r = fut_r[np.isfinite(fut_r)]
                if len(fut_r) > 0:
                    future_min[i] = fut_r.min()
        grp["future_min_rating"] = future_min
        rows.append(grp)
    panel = pd.concat(rows, ignore_index=True)
    panel["downgrade"] = (panel["future_min_rating"] < panel["rating_numeric"]).astype(int)

    # Price crash from trades
    trades = trades.sort_values(["isin", "date"]).copy()
    trades["close_prev"] = trades.groupby("isin")["close_price"].shift(1)
    trades["log_p"] = np.log(trades["close_price"])
    # Forward min close in next 30 days
    rows2 = []
    for isin, grp in trades.groupby("isin"):
        grp = grp.sort_values("date").reset_index(drop=True)
        prices = grp["close_price"].to_numpy()
        dates = grp["date"].to_numpy()
        fwd_min = np.full(len(grp), np.nan)
        for i in range(len(grp)):
            if not np.isfinite(prices[i]) or prices[i] <= 0:
                continue
            end_date = dates[i] + np.timedelta64(PRICE_DROP_WINDOW_DAYS, 'D')
            future_idx = np.where((dates > dates[i]) & (dates <= end_date))[0]
            if len(future_idx) > 0:
                fp = prices[future_idx]
                fp = fp[np.isfinite(fp) & (fp > 0)]
                if len(fp) > 0:
                    fwd_min[i] = fp.min()
        grp["fwd_min_30d"] = fwd_min
        rows2.append(grp[["date", "isin", "close_price", "fwd_min_30d"]])
    crash_df = pd.concat(rows2, ignore_index=True)
    crash_df["fwd_drop"] = (crash_df["fwd_min_30d"] / crash_df["close_price"] - 1)
    crash_df["crash"] = (crash_df["fwd_drop"] < PRICE_DROP_THRESHOLD).astype(int)

    panel = panel.merge(crash_df[["date", "isin", "crash"]], on=["date", "isin"], how="left")
    panel["crash"] = panel["crash"].fillna(0).astype(int)
    panel["target_adverse"] = ((panel["downgrade"] == 1) | (panel["crash"] == 1)).astype(int)
    return panel


def walk_forward_split(df: pd.DataFrame, train_months: int = 12, step_months: int = 1):
    dates = sorted(df["date"].dt.to_period("M").unique())
    for i in range(train_months, len(dates), step_months):
        train_periods = dates[max(0, i - train_months):i]
        test_period = dates[i]
        train_mask = df["date"].dt.to_period("M").isin(train_periods)
        test_mask = df["date"].dt.to_period("M") == test_period
        yield df[train_mask], df[test_mask]


def train_catboost_classifier(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    print("\n=== CatBoost classifier ===")
    folds = list(walk_forward_split(df))
    print(f"Total folds: {len(folds)}")
    all_preds = []
    fold_metrics = []

    cat_idx = [FEATURES_ALL.index(f) for f in CAT_FEATURES]
    t0 = time.time()
    for i, (train_df, test_df) in enumerate(folds, start=1):
        train_df = train_df.dropna(subset=["target_adverse"]).copy()
        test_df = test_df.dropna(subset=["target_adverse"]).copy()
        if len(train_df) < 100 or len(test_df) < 10:
            continue
        X_tr = train_df[FEATURES_ALL].copy()
        X_te = test_df[FEATURES_ALL].copy()
        for c in CAT_FEATURES:
            X_tr[c] = X_tr[c].fillna("Unknown").astype(str)
            X_te[c] = X_te[c].fillna("Unknown").astype(str)
        y_tr = train_df["target_adverse"]
        y_te = test_df["target_adverse"]
        if y_tr.sum() < 5:
            continue  # too few positives

        model = CatBoostClassifier(
            depth=6, iterations=500, learning_rate=0.05,
            cat_features=cat_idx, verbose=0, random_seed=42,
            allow_writing_files=False, thread_count=-1,
        )
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]

        preds = test_df[["date", "isin"]].copy()
        preds["default_proba"] = proba
        all_preds.append(preds)

        try:
            auc = roc_auc_score(y_te, proba) if y_te.sum() > 0 else np.nan
        except Exception:
            auc = np.nan
        try:
            ap = average_precision_score(y_te, proba) if y_te.sum() > 0 else np.nan
        except Exception:
            ap = np.nan
        fold_metrics.append({"fold": i, "n_test": len(test_df), "n_pos": int(y_te.sum()),
                              "AUC": auc, "AP": ap})

        if i % 10 == 0 or i == len(folds):
            print(f"  fold {i:3d}/{len(folds)}: n_pos={int(y_te.sum())}, AUC={auc:.3f}, AP={ap:.3f}, "
                  f"elapsed {time.time() - t0:.0f}s")

    preds_df = pd.concat(all_preds, ignore_index=True)
    return preds_df, fold_metrics


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden=(64, 32, 16), dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_mlp_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CAT_FEATURES),
    ], remainder="drop")


def train_mlp_classifier(df: pd.DataFrame, max_epochs: int = 100, patience: int = 10) -> tuple[pd.DataFrame, list]:
    print("\n=== MLP (PyTorch) classifier ===")
    folds = list(walk_forward_split(df))
    print(f"Total folds: {len(folds)}")
    all_preds = []
    fold_metrics = []
    device = torch.device("cpu")
    t0 = time.time()

    for i, (train_df, test_df) in enumerate(folds, start=1):
        train_df = train_df.dropna(subset=["target_adverse"]).copy()
        test_df = test_df.dropna(subset=["target_adverse"]).copy()
        if len(train_df) < 100 or len(test_df) < 10:
            continue
        for c in CAT_FEATURES:
            train_df[c] = train_df[c].fillna("Unknown").astype(str)
            test_df[c] = test_df[c].fillna("Unknown").astype(str)
        y_tr = train_df["target_adverse"].to_numpy(dtype=np.float32)
        y_te = test_df["target_adverse"].to_numpy(dtype=np.float32)
        if y_tr.sum() < 5:
            continue

        prep = build_mlp_preprocessor()
        X_tr = prep.fit_transform(train_df[FEATURES_ALL])
        X_te = prep.transform(test_df[FEATURES_ALL])

        # Class weights to handle imbalance
        pos_w = float((1 - y_tr.mean()) / max(y_tr.mean(), 1e-6))
        pos_w_tensor = torch.tensor([pos_w], dtype=torch.float32, device=device)

        Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
        yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
        Xv = torch.tensor(X_te, dtype=torch.float32, device=device)

        model = MLPClassifier(in_dim=X_tr.shape[1]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        best_loss = float("inf")
        no_improve = 0
        # 80/20 train/val split for early stopping (within train fold, по времени)
        n_tr = int(0.8 * len(Xt))
        Xt_train, Xt_val = Xt[:n_tr], Xt[n_tr:]
        yt_train, yt_val = yt[:n_tr], yt[n_tr:]

        for epoch in range(max_epochs):
            model.train()
            opt.zero_grad()
            logits = model(Xt_train)
            loss = criterion(logits, yt_train)
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(Xt_val)
                val_loss = criterion(val_logits, yt_val).item()
            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

        model.eval()
        with torch.no_grad():
            test_logits = model(Xv)
            proba = torch.sigmoid(test_logits).cpu().numpy()

        preds = test_df[["date", "isin"]].copy()
        preds["default_proba"] = proba
        all_preds.append(preds)

        try:
            auc = roc_auc_score(y_te, proba) if y_te.sum() > 0 else np.nan
        except Exception:
            auc = np.nan
        try:
            ap = average_precision_score(y_te, proba) if y_te.sum() > 0 else np.nan
        except Exception:
            ap = np.nan
        fold_metrics.append({"fold": i, "n_test": len(test_df), "n_pos": int(y_te.sum()),
                              "AUC": auc, "AP": ap})

        if i % 10 == 0 or i == len(folds):
            print(f"  fold {i:3d}/{len(folds)}: n_pos={int(y_te.sum())}, AUC={auc:.3f}, AP={ap:.3f}, "
                  f"elapsed {time.time() - t0:.0f}s")

    preds_df = pd.concat(all_preds, ignore_index=True)
    return preds_df, fold_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="catboost,mlp")
    parser.add_argument("--quick", action="store_true", help="First 12 folds only")
    args = parser.parse_args()

    print("Loading data...")
    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])
    panel["sector"] = panel["sector"].fillna("Unknown").astype(str)
    panel.loc[panel["sector"].str.strip() == "", "sector"] = "Unknown"
    trades = pd.read_csv(TRADES_PATH, parse_dates=["date"], usecols=["date", "isin", "close_price"])
    print(f"Panel: {panel.shape}, trades: {trades.shape}")

    print("Building target...")
    panel = build_target(panel, trades)
    print(f"Positive rate (adverse event in next 180d): {panel['target_adverse'].mean():.3%}")
    print(f"  downgrades: {panel['downgrade'].sum():,}")
    print(f"  crashes:    {panel['crash'].sum():,}")
    print(f"  combined:   {panel['target_adverse'].sum():,}")

    models = [m.strip() for m in args.models.split(",")]
    stats = []

    if "catboost" in models:
        preds_cb, m_cb = train_catboost_classifier(panel)
        preds_cb.to_csv(DATA_DIR / "credit_risk_predictions_cb.csv", index=False)
        pd.DataFrame(m_cb).to_csv(DATA_DIR / "credit_risk_folds_cb.csv", index=False)
        if m_cb:
            ms = pd.DataFrame(m_cb)
            stats.append({"model": "catboost", "n_folds": len(ms),
                          "AUC_mean": ms["AUC"].mean(), "AUC_med": ms["AUC"].median(),
                          "AP_mean": ms["AP"].mean()})
            print(f"\nCatBoost overall: AUC mean={ms['AUC'].mean():.3f}, median={ms['AUC'].median():.3f}")

    if "mlp" in models:
        preds_mlp, m_mlp = train_mlp_classifier(panel)
        preds_mlp.to_csv(DATA_DIR / "credit_risk_predictions_mlp.csv", index=False)
        pd.DataFrame(m_mlp).to_csv(DATA_DIR / "credit_risk_folds_mlp.csv", index=False)
        if m_mlp:
            ms = pd.DataFrame(m_mlp)
            stats.append({"model": "mlp", "n_folds": len(ms),
                          "AUC_mean": ms["AUC"].mean(), "AUC_med": ms["AUC"].median(),
                          "AP_mean": ms["AP"].mean()})
            print(f"\nMLP overall: AUC mean={ms['AUC'].mean():.3f}, median={ms['AUC'].median():.3f}")

    pd.DataFrame(stats).to_csv(DATA_DIR / "credit_risk_summary.csv", index=False)
    print("\n=== Summary ===")
    print(pd.DataFrame(stats).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
