"""Honest strict-OOS evaluation всех Stage 6 variants на clean holdout.

Holdout window: 2025-10-01 .. 2026-03-30 (6 мес, 123 trading days).
Все variants applied with SAME fixed config (как разрабатывались на main),
прогоняются на holdout БЕЗ переоптимизации.

Сравнение vs OBLG, SBRB, TBRU (доступные retail ETFs) и RUCBTRNS (theoretical).

Output: data/honest_holdout_results.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

import scripts.strategy_lib as sl

HOLDOUT_START = pd.Timestamp("2025-10-01")
HOLDOUT_END = pd.Timestamp("2026-03-30")
CAPITAL = 10_000_000


def load_sig(model: str = "no_lag") -> pd.DataFrame:
    sig = sl.load_signal_full(model)
    sig_wl = sl.load_signal_full("with_lag")
    wl = sig_wl[["date", "isin", "mispricing_z"]].rename(
        columns={"mispricing_z": "mispricing_z_wl"})
    sig = sig.merge(wl, on=["date", "isin"], how="inner")
    sig["mispricing_z_nl"] = sig["mispricing_z"]
    sig["mispricing_avg_z"] = (sig["mispricing_z_nl"] + sig["mispricing_z_wl"]) / 2
    sig = sl.compute_quality_score(sig)
    return sig


def macro_overlay(rets: pd.DataFrame) -> pd.DataFrame:
    """Apply V13a-style rules-based overlay."""
    macro = pd.read_csv(sl.DATA_DIR / "macro.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    macro["rvi_p90"] = macro["rvi"].rolling(252, min_periods=60).quantile(0.90)
    macro["s1"] = (macro["rvi"] > macro["rvi_p90"]).fillna(False).astype(int)
    macro["s2"] = (macro["key_rate"].diff(5).abs() > 2.0).fillna(False).astype(int)
    macro["s3"] = (macro["imoex_close"].pct_change(5) < -0.10).fillna(False).astype(int)
    macro["s4"] = (macro["rvi"] > 50).astype(int)
    macro["stress_idx"] = macro["s1"] + macro["s2"] + macro["s3"] + macro["s4"]
    macro["regime"] = np.where(macro["stress_idx"] >= 2, "stress", "normal")
    out = rets.merge(macro[["date", "regime"]], on="date", how="left")
    out["regime"] = out["regime"].fillna("normal")
    out["return"] = np.where(out["regime"] == "stress", 0.0, out["return"])
    return out[["date", "return", "n_positions"]]


def build_baseline_predictions_signal(model_kind: str) -> pd.DataFrame:
    """Используем predictions из OLS / Ridge / no_lag CatBoost.

    Каждая модель trained walk-forward → predictions strict OOS per date.
    """
    if model_kind == "naive":
        sig = sl.load_signal_full("no_lag")
        sig["mispricing"] = sig["g_spread_actual"]  # raw g_spread = carry signal
    elif model_kind in ("ols", "ridge"):
        sig = sl.load_signal_full("no_lag")
        bp = pd.read_csv(sl.DATA_DIR / f"predictions_{model_kind}.csv", parse_dates=["date"])
        bp = bp[["date", "isin", "g_spread_predicted"]].rename(
            columns={"g_spread_predicted": "g_spread_predicted_baseline"})
        sig = sig.merge(bp, on=["date", "isin"], how="left")
        sig["g_spread_predicted"] = sig["g_spread_predicted_baseline"].fillna(sig["g_spread_predicted"])
        sig = sig.drop(columns=["g_spread_predicted_baseline"])
        sig["mispricing"] = sig["g_spread_actual"] - sig["g_spread_predicted"]
    elif model_kind == "catboost":
        sig = sl.load_signal_full("no_lag")
        # mispricing уже = actual - catboost_predicted
    else:
        raise ValueError(f"unknown model_kind: {model_kind}")
    sig = sig.drop(columns=["mispricing_z", "mispricing_quintile"])
    sig = sl._compute_cs_z_and_quintile(sig, "mispricing")
    sig = sl.compute_quality_score(sig)
    return sig


def evaluate_variant(rets: pd.DataFrame, name: str) -> dict:
    holdout = rets[(rets["date"] >= HOLDOUT_START) & (rets["date"] <= HOLDOUT_END)].copy()
    if len(holdout) == 0:
        return {"name": name, "n_days": 0}
    m = sl.calc_metrics(holdout, capital=CAPITAL, name=name)
    return m


def run_strategy_variant(
    sig: pd.DataFrame,
    n_max: int = 25,
    freq: str = "Q",
    ranking_col: str = "mispricing",
    use_quality: bool = False,
    sizing: str = "equal",
    use_overlay: bool = False,
) -> pd.DataFrame:
    mask = sig["quality_z"] > 0 if use_quality else None
    rets, _ = sl.backtest_rule_c(
        sig, n_max=n_max, freq=freq, ranking_col=ranking_col,
        filter_mask=mask, sizing=sizing,
    )
    if use_overlay:
        rets = macro_overlay(rets)
    return rets


def main() -> None:
    results = []

    print("=" * 80)
    print("HONEST HOLDOUT EVALUATION (2025-10-01 .. 2026-03-30, strict OOS)")
    print("=" * 80)

    # ---- Baselines (different mispricing models) ----
    print("\n--- Baselines: different prediction models, monthly equal, no quality ---")
    for mk in ["naive", "ols", "ridge", "catboost"]:
        sig = build_baseline_predictions_signal(mk)
        rets = run_strategy_variant(sig, freq="M", ranking_col="mispricing")
        m = evaluate_variant(rets, f"{mk}_baseline_M_eq")
        m["variant_group"] = "baseline_models"
        m["model"] = mk
        results.append(m)
        print(f"  {mk:10s} M+EQ:  Sharpe={m.get('Sharpe', float('nan')):.3f}, "
              f"profit={m.get('profit_RUB', 0):>12,.0f}")

    # ---- Quality filter variants (CatBoost) ----
    print("\n--- Quality filter variants (CatBoost no-lag, monthly) ---")
    sig_cat = load_sig("no_lag")
    for label, mask_thresh in [("V0_vanilla", None), ("V1_Q>0", 0.0), ("V2_Q>0.5", 0.5)]:
        mask = (sig_cat["quality_z"] > mask_thresh) if mask_thresh is not None else None
        rets, _ = sl.backtest_rule_c(sig_cat, n_max=25, freq="M",
                                      ranking_col="mispricing", filter_mask=mask)
        m = evaluate_variant(rets, label)
        m["variant_group"] = "quality_filter"
        results.append(m)
        print(f"  {label:15s}: Sharpe={m.get('Sharpe', float('nan')):.3f}, "
              f"profit={m.get('profit_RUB', 0):>12,.0f}")

    # ---- Construction variants (V10x family) ----
    print("\n--- Construction overlays (на foundation Q>0) ---")
    sig_full = load_sig("no_lag")  # has mispricing_avg_z too
    for label, freq, sn, sz, use_avg in [
        ("V10a_M_SN", "M", True, "equal", False),
        ("V10b_M_zw", "M", False, "z_weighted", False),
        ("V10c_Q_eq", "Q", False, "equal", False),
        ("V10d_Q_SN", "Q", True, "equal", False),
        ("V10e_Q_zw", "Q", False, "z_weighted", False),
        ("V10f_Q_SN_zw", "Q", True, "z_weighted", False),
    ]:
        mask = sig_full["quality_z"] > 0
        rets, _ = sl.backtest_rule_c(
            sig_full, n_max=25, freq=freq,
            ranking_col="mispricing_avg_z" if use_avg else "mispricing",
            filter_mask=mask, sector_neutral=sn, n_per_sector=2, sizing=sz,
        )
        m = evaluate_variant(rets, label)
        m["variant_group"] = "construction"
        results.append(m)
        print(f"  {label:15s}: Sharpe={m.get('Sharpe', float('nan')):.3f}, "
              f"profit={m.get('profit_RUB', 0):>12,.0f}")

    # ---- Ensemble variants (V11x family) ----
    print("\n--- Ensemble variants ---")
    for label, freq, rank, sz in [
        ("V11a_avgz_M", "M", "mispricing_avg_z", "equal"),
        ("V11e_avgz_Q_zw", "Q", "mispricing_avg_z", "z_weighted"),
        ("V11f_avgz_Q_eq", "Q", "mispricing_avg_z", "equal"),
    ]:
        mask = sig_full["quality_z"] > 0
        rets, _ = sl.backtest_rule_c(
            sig_full, n_max=25, freq=freq, ranking_col=rank,
            filter_mask=mask, sizing=sz,
        )
        m = evaluate_variant(rets, label)
        m["variant_group"] = "ensemble"
        results.append(m)
        print(f"  {label:18s}: Sharpe={m.get('Sharpe', float('nan')):.3f}, "
              f"profit={m.get('profit_RUB', 0):>12,.0f}")

    # ---- Overlay variants (V13x family) ----
    print("\n--- Macro overlay variants (на V11e foundation) ---")
    mask = sig_full["quality_z"] > 0
    rets_v11e, _ = sl.backtest_rule_c(
        sig_full, n_max=25, freq="Q", ranking_col="mispricing_avg_z",
        filter_mask=mask, sizing="z_weighted",
    )
    # V11e (no overlay)
    m = evaluate_variant(rets_v11e, "V11e_no_overlay")
    m["variant_group"] = "overlay_baseline"
    results.append(m)
    print(f"  V11e (no overlay):  Sharpe={m.get('Sharpe', float('nan')):.3f}, "
          f"profit={m.get('profit_RUB', 0):>12,.0f}")
    # V13a (rules cut 0%)
    rets_v13a = macro_overlay(rets_v11e)
    m = evaluate_variant(rets_v13a, "V13a_rules_overlay")
    m["variant_group"] = "overlay"
    results.append(m)
    print(f"  V13a (rules overlay): Sharpe={m.get('Sharpe', float('nan')):.3f}, "
          f"profit={m.get('profit_RUB', 0):>12,.0f}")
    # V13b (50% scale)
    rets_v13b = rets_v11e.copy()
    macro = pd.read_csv(sl.DATA_DIR / "macro.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    macro["rvi_p90"] = macro["rvi"].rolling(252, min_periods=60).quantile(0.90)
    macro["s1"] = (macro["rvi"] > macro["rvi_p90"]).fillna(False).astype(int)
    macro["s2"] = (macro["key_rate"].diff(5).abs() > 2.0).fillna(False).astype(int)
    macro["s3"] = (macro["imoex_close"].pct_change(5) < -0.10).fillna(False).astype(int)
    macro["s4"] = (macro["rvi"] > 50).astype(int)
    macro["stress_idx"] = macro["s1"] + macro["s2"] + macro["s3"] + macro["s4"]
    macro["regime"] = np.where(macro["stress_idx"] >= 2, "stress", "normal")
    rets_v13b = rets_v13b.merge(macro[["date", "regime"]], on="date", how="left")
    rets_v13b["regime"] = rets_v13b["regime"].fillna("normal")
    rets_v13b["return"] = np.where(rets_v13b["regime"] == "stress", rets_v13b["return"] * 0.5, rets_v13b["return"])
    rets_v13b = rets_v13b[["date", "return", "n_positions"]]
    m = evaluate_variant(rets_v13b, "V13b_50pct_overlay")
    m["variant_group"] = "overlay"
    results.append(m)
    print(f"  V13b (50% overlay):   Sharpe={m.get('Sharpe', float('nan')):.3f}, "
          f"profit={m.get('profit_RUB', 0):>12,.0f}")

    print("\n--- Benchmarks on same holdout ---")
    etfs = pd.read_csv(sl.DATA_DIR / "etfs.csv", parse_dates=["date"])
    bench = sl.load_benchmarks()
    for etf, col in [("OBLG", "oblg_ret"), ("SBRB", "sbrb_ret"), ("TBRU", "tbru_ret")]:
        df = etfs[["date", col]].dropna().rename(columns={col: "return"})
        df["n_positions"] = 1
        m = evaluate_variant(df, etf)
        m["variant_group"] = "ETF"
        results.append(m)
        print(f"  {etf:8s}: Sharpe={m.get('Sharpe', float('nan')):.3f}, "
              f"profit={m.get('profit_RUB', 0):>12,.0f}")
    df = bench[["date", "rucbtrns_ret"]].dropna().rename(columns={"rucbtrns_ret": "return"})
    df["n_positions"] = 1
    m = evaluate_variant(df, "RUCBTRNS")
    m["variant_group"] = "index_theoretical"
    results.append(m)
    print(f"  {'RUCBTRNS':8s}: Sharpe={m.get('Sharpe', float('nan')):.3f}, "
          f"profit={m.get('profit_RUB', 0):>12,.0f}")

    df_out = pd.DataFrame(results)
    df_out.to_csv(sl.DATA_DIR / "honest_holdout_results.csv", index=False)
    print(f"\nSaved -> {sl.DATA_DIR / 'honest_holdout_results.csv'}")
    print(f"Total variants evaluated: {len(df_out)}")

    print("\n" + "=" * 80)
    print("FINAL TABLE (sorted by profit)")
    print("=" * 80)
    cols = ["name", "variant_group", "Sharpe", "CAGR", "MaxDD", "profit_RUB"]
    sorted_df = df_out.sort_values("profit_RUB", ascending=False)
    print(sorted_df[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
