"""Сравнение двух моделей: no_lag vs with_lag в Rule C (monthly re-rank).

Запускает backtest на обеих моделях и выдаёт сравнение в деньгах.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MAIN_END = pd.Timestamp("2025-09-30")
HOLDOUT_START = pd.Timestamp("2025-10-01")
N_MAX = 25
INITIAL_CAPITAL = 10_000_000  # 10M RUB


def build_signal(pred_path: Path, panel_path: Path, trades_path: Path) -> pd.DataFrame:
    """Loaded predictions + panel + trades -> signal с total_return."""
    preds = pd.read_csv(pred_path, parse_dates=["date"])
    preds["mispricing"] = preds["g_spread_actual"] - preds["g_spread_predicted"]

    panel = pd.read_csv(panel_path, parse_dates=["date"],
                        usecols=["date", "isin", "volume_ma20", "coupon_rate"])

    trades = pd.read_csv(trades_path, parse_dates=["date"],
                         usecols=["date", "isin", "close_price"])

    df = preds.merge(panel, on=["date", "isin"], how="left")
    df = df.merge(trades, on=["date", "isin"], how="left")
    df = df.sort_values(["isin", "date"]).reset_index(drop=True)

    df["close_prev"] = df.groupby("isin")["close_price"].shift(1)
    df["price_return"] = df["close_price"] / df["close_prev"] - 1
    df.loc[df["price_return"].abs() > 0.20, "price_return"] = np.nan
    df["coupon_daily"] = df["coupon_rate"].fillna(0) / 100 / 365
    df["total_return"] = df["price_return"].fillna(0) + df["coupon_daily"]
    df["tradable"] = df["volume_ma20"].fillna(0) > 5_000_000

    return df


def rule_c_backtest(sig: pd.DataFrame, n_max: int = N_MAX,
                    cost_rt: float = 0.005) -> pd.DataFrame:
    """Rule C: monthly re-rank. Returns daily portfolio returns."""
    sig = sig.copy()
    sig["month"] = sig["date"].dt.to_period("M")

    selections = []
    for month, grp in sig.groupby("month"):
        last_date = grp["date"].max()
        day_df = grp[grp["date"] == last_date]
        eligible = day_df[day_df["tradable"]].sort_values("mispricing", ascending=False).head(n_max)
        selections.append({"rebalance_date": last_date, "isins": eligible["isin"].tolist()})

    sels = pd.DataFrame(selections).sort_values("rebalance_date").reset_index(drop=True)
    sig_idx = sig.set_index(["date", "isin"])
    all_dates = np.sort(sig["date"].unique())

    daily_rows = []
    for idx in range(len(sels) - 1):
        start = sels.iloc[idx]["rebalance_date"]
        end = sels.iloc[idx + 1]["rebalance_date"]
        isins = sels.iloc[idx]["isins"]
        if not isins:
            continue
        period_dates = [d for d in all_dates if start < d <= end]
        avg_cost = cost_rt / 2  # per leg
        for d in period_dates:
            rets = []
            for isin in isins:
                key = (d, isin)
                if key in sig_idx.index:
                    rets.append(sig_idx.loc[key, "total_return"])
            if not rets:
                continue
            r = np.mean(rets)
            if d == period_dates[0]:
                r -= avg_cost
            if d == period_dates[-1]:
                r -= avg_cost
            daily_rows.append({"date": d, "return": r, "n_positions": len(isins)})
    return pd.DataFrame(daily_rows)


def metrics(rets: pd.DataFrame, capital: float = INITIAL_CAPITAL) -> dict:
    if len(rets) == 0:
        return {}
    r = rets["return"].values
    eq = np.cumprod(1 + r)
    total_ret = eq[-1] - 1
    n_years = len(r) / 252
    cagr = eq[-1] ** (1 / n_years) - 1 if n_years > 0 else np.nan
    sharpe = np.sqrt(252) * r.mean() / r.std() if r.std() > 0 else np.nan
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = dd.min()
    end_capital = capital * eq[-1]
    profit = end_capital - capital
    return {
        "n_days": len(r),
        "n_years": n_years,
        "total_return": total_ret,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "end_capital_RUB": end_capital,
        "profit_RUB": profit,
        "profit_per_year_RUB": profit / max(n_years, 0.001),
    }


def report(name: str, rets: pd.DataFrame, capital: float = INITIAL_CAPITAL):
    m = metrics(rets, capital)
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")
    print(f"  Период              : {m['n_years']:.1f} лет ({m['n_days']} торговых дней)")
    print(f"  Total return        : {m['total_return'] * 100:+.2f}%")
    print(f"  CAGR                : {m['CAGR'] * 100:+.2f}%")
    print(f"  Sharpe              : {m['Sharpe']:+.3f}")
    print(f"  MaxDD               : {m['MaxDD'] * 100:+.2f}%")
    print(f"  Стартовый капитал   : {capital:,.0f} RUB")
    print(f"  Конечный капитал    : {m['end_capital_RUB']:,.0f} RUB")
    print(f"  Прибыль             : {m['profit_RUB']:+,.0f} RUB")
    print(f"  Прибыль в год       : {m['profit_per_year_RUB']:+,.0f} RUB/год")


def main():
    print("Загрузка no_lag и with_lag сигналов...")
    sig_no = build_signal(DATA / "predictions_no_lag.csv",
                          DATA / "panel_engineered.csv",
                          DATA / "trades_daily.csv")
    sig_wl = build_signal(DATA / "predictions_with_lag.csv",
                          DATA / "panel_engineered.csv",
                          DATA / "trades_daily.csv")

    print(f"no_lag : {sig_no.shape}, bonds={sig_no['isin'].nunique()}")
    print(f"with_lag: {sig_wl.shape}, bonds={sig_wl['isin'].nunique()}")

    print("\nBacktesting Rule C (no_lag)...")
    rets_no = rule_c_backtest(sig_no)
    print("Backtesting Rule C (with_lag)...")
    rets_wl = rule_c_backtest(sig_wl)

    # Split main / holdout
    def split(r):
        return r[r["date"] <= MAIN_END], r[r["date"] >= HOLDOUT_START]

    no_main, no_hold = split(rets_no)
    wl_main, wl_hold = split(rets_wl)

    print("\n\n" + "#" * 70)
    print("#  СРАВНЕНИЕ ДВУХ МОДЕЛЕЙ (Rule C, 10M RUB начальный капитал)")
    print("#" * 70)

    report("Модель 1: no_lag (R^2=0.524) — Main 2021-01..2025-09", no_main)
    report("Модель 2: with_lag (R^2=0.796) — Main 2021-01..2025-09", wl_main)
    report("Модель 1: no_lag — Clean Holdout 2025-10..2026-03", no_hold)
    report("Модель 2: with_lag — Clean Holdout 2025-10..2026-03", wl_hold)

    print("\n\n" + "#" * 70)
    print("#  Бенчмарк: passive equal-weight tradable (без сигнала)")
    print("#" * 70)
    bench_no = sig_no[sig_no["tradable"]].groupby("date").agg(
        return_=("total_return", "mean")).reset_index().rename(columns={"return_": "return"})
    bm_main, bm_hold = split(bench_no)
    report("Бенчмарк EW — Main", bm_main)
    report("Бенчмарк EW — Holdout", bm_hold)

    full = []
    for label, rets in [
        ("no_lag (Main)", no_main),
        ("no_lag (Holdout)", no_hold),
        ("with_lag (Main)", wl_main),
        ("with_lag (Holdout)", wl_hold),
        ("Benchmark EW (Main)", bm_main),
        ("Benchmark EW (Holdout)", bm_hold),
    ]:
        m = metrics(rets)
        m["strategy"] = label
        full.append(m)
    out = pd.DataFrame(full)[["strategy", "n_days", "n_years", "total_return",
                              "CAGR", "Sharpe", "MaxDD",
                              "end_capital_RUB", "profit_RUB", "profit_per_year_RUB"]]
    out.to_csv(DATA / "backtest_models_compare.csv", index=False)
    print(f"\nSaved -> {DATA / 'backtest_models_compare.csv'}")


if __name__ == "__main__":
    main()
