"""Разложение: чем именно отличаются Q5-портфели от benchmark и где теряем."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MAIN_END = pd.Timestamp("2025-09-30")
HOLDOUT_START = pd.Timestamp("2025-10-01")


def load() -> pd.DataFrame:
    preds = pd.read_csv(DATA / "predictions_no_lag.csv", parse_dates=["date"])
    preds["mispricing"] = preds["g_spread_actual"] - preds["g_spread_predicted"]

    panel = pd.read_csv(DATA / "panel_engineered.csv", parse_dates=["date"],
                        usecols=["date", "isin", "volume_ma20", "coupon_rate",
                                 "rating_numeric", "duration", "time_to_maturity",
                                 "sector", "g_spread"])
    trades = pd.read_csv(DATA / "trades_daily.csv", parse_dates=["date"],
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

    # Cross-sectional quintile by mispricing per date
    def _q(s):
        if s.notna().sum() < 5:
            return pd.Series([np.nan] * len(s), index=s.index)
        return pd.qcut(s.rank(method="first"), 5, labels=False) + 1
    df["quintile"] = df.groupby("date")["mispricing"].transform(_q)
    return df


def compare_q5_vs_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Compare avg characteristics: Q5 (tradable) vs full tradable universe."""
    df = df[df["date"] <= MAIN_END].copy()
    q5 = df[(df["quintile"] == 5) & (df["tradable"])]
    bm = df[df["tradable"]]

    rows = []
    for label, sub in [("Q5 (стратегия)", q5), ("All tradable (бенч)", bm)]:
        rows.append({
            "Группа": label,
            "Bonds-days": len(sub),
            "Avg coupon %": sub["coupon_rate"].mean(),
            "Avg rating": sub["rating_numeric"].mean(),  # higher = better
            "Avg duration": sub["duration"].mean(),
            "Avg g_spread": sub["g_spread"].mean(),
            "Avg price ret/day %": sub["price_return"].mean() * 100,
            "Avg coupon/day %": sub["coupon_daily"].mean() * 100,
            "Avg total ret/day %": sub["total_return"].mean() * 100,
            "Std total ret/day %": sub["total_return"].std() * 100,
        })
    return pd.DataFrame(rows)


def decompose_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Sources of return: price (capital gains) vs coupon (carry)."""
    df = df[df["date"] <= MAIN_END].copy()
    q5 = df[(df["quintile"] == 5) & (df["tradable"])]
    bm = df[df["tradable"]]

    rows = []
    for label, sub in [("Q5", q5), ("Benchmark", bm)]:
        # Annualized contribution
        price_ann = sub["price_return"].mean() * 252
        coupon_ann = sub["coupon_daily"].mean() * 252
        total_ann = sub["total_return"].mean() * 252
        rows.append({
            "Источник": label,
            "Price return ann %": price_ann * 100,
            "Coupon ann %": coupon_ann * 100,
            "Total ann %": total_ann * 100,
            "Price share %": price_ann / total_ann * 100 if total_ann else 0,
            "Coupon share %": coupon_ann / total_ann * 100 if total_ann else 0,
        })
    return pd.DataFrame(rows)


def sector_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["date"] <= MAIN_END].copy()
    q5 = df[(df["quintile"] == 5) & (df["tradable"])]
    bm = df[df["tradable"]]

    q5_sec = q5["sector"].value_counts(normalize=True).head(10) * 100
    bm_sec = bm["sector"].value_counts(normalize=True).head(10) * 100
    res = pd.DataFrame({"Q5 %": q5_sec, "Bench %": bm_sec}).fillna(0)
    res["diff"] = res["Q5 %"] - res["Bench %"]
    return res.sort_values("diff", ascending=False)


def rating_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Rating buckets: IG (>=6) vs HY (<6) vs unrated (NaN)."""
    df = df[df["date"] <= MAIN_END].copy()
    q5 = df[(df["quintile"] == 5) & (df["tradable"])]
    bm = df[df["tradable"]]

    def bucket(s):
        out = pd.Series("Unrated", index=s.index)
        out[s >= 12] = "AAA (>=12)"
        out[(s >= 9) & (s < 12)] = "AA-A (9-11)"
        out[(s >= 6) & (s < 9)] = "BBB (6-8)"
        out[(s >= 3) & (s < 6)] = "BB (3-5)"
        out[s < 3] = "B и ниже (<3)"
        return out

    q5_b = bucket(q5["rating_numeric"]).value_counts(normalize=True) * 100
    bm_b = bucket(bm["rating_numeric"]).value_counts(normalize=True) * 100
    res = pd.DataFrame({"Q5 %": q5_b, "Bench %": bm_b}).fillna(0)
    res["diff"] = res["Q5 %"] - res["Bench %"]
    return res.sort_index()


def main():
    df = load()
    print(f"Loaded: {df.shape}, bonds={df['isin'].nunique()}")
    print(f"Period: {df['date'].min().date()} .. {df['date'].max().date()}")
    print()

    print("=" * 70)
    print("1. Сравнение характеристик: Q5 vs Benchmark (Main period)")
    print("=" * 70)
    cmp = compare_q5_vs_benchmark(df)
    print(cmp.to_string(index=False))

    print()
    print("=" * 70)
    print("2. Декомпозиция returns: price gains vs coupon carry")
    print("=" * 70)
    dec = decompose_returns(df)
    print(dec.to_string(index=False))

    print()
    print("=" * 70)
    print("3. Распределение по секторам (top 10)")
    print("=" * 70)
    print(sector_breakdown(df).round(2).to_string())

    print()
    print("=" * 70)
    print("4. Распределение по рейтингам")
    print("=" * 70)
    print(rating_distribution(df).round(2).to_string())


if __name__ == "__main__":
    main()
