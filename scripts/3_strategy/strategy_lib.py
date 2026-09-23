"""Reusable backtest framework for Stage 6 strategy improvements (NB 09-15).

API summary:
    load_signal_full(predictions='no_lag') -> DataFrame
        Полностью обогащённый dataframe: predictions + mispricing + z + quintile +
        coupon/price returns + tradable + close_price + roll_spread.

    compute_quality_score(df, weights=None, ...) -> DataFrame
        Добавляет colonne `quality_z` (cross-sectional z-score композита).

    compute_momentum(df, lookback_days=30) -> DataFrame
        Добавляет `mom_return_<L>d` и `mom_z`.

    backtest_rule_c(df, n_max=25, freq='M', ranking_col='mispricing',
                    filter_mask=None, name='') -> tuple[DataFrame, DataFrame]
        Возвращает (daily_returns, trades).

    calc_metrics(rets, capital=10_000_000) -> dict
        Sharpe, Sortino, MaxDD, CAGR, profit, etc.

    block_bootstrap_sharpe(rets, n_boot=1000, block=60) -> ndarray

    compare_variants({name: rets_df}) -> DataFrame
        Side-by-side метрики.

    split_main_holdout(rets) -> (main, holdout)

Constants:
    DATA_DIR, MAIN_END, HOLDOUT_START, N_MAX_DEFAULT, LIQUIDITY_MIN_RUB

Все функции pure-ish (no mutation of args). Type hints, msgspec не нужен (CSV).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

MAIN_END = pd.Timestamp("2025-09-30")
HOLDOUT_START = pd.Timestamp("2025-10-01")

N_MAX_DEFAULT = 25
LIQUIDITY_MIN_RUB = 5_000_000
ROLL_FALLBACK_BPS = 0.0050  # 50 bps round-trip fallback
ROLL_MIN_BPS = 0.0010
ROLL_MAX_BPS = 0.0500
DEFAULT_CAPITAL = 10_000_000

ModelKind = Literal["no_lag", "with_lag"]


def _load_predictions(kind: ModelKind) -> pd.DataFrame:
    path = DATA_DIR / f"predictions_{kind}.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df["mispricing"] = df["g_spread_actual"] - df["g_spread_predicted"]
    return df


def _load_panel_subset() -> pd.DataFrame:
    cols = [
        "date", "isin", "g_spread", "duration", "rating_numeric", "sector",
        "volume_ma20", "coupon_rate", "coupon_frequency", "time_to_maturity",
        "net_debt_ebitda", "debt_to_equity", "roa", "roe",
        "rsbu_icr", "ebitda_margin", "issuer_name",
    ]
    df = pd.read_csv(DATA_DIR / "panel_engineered.csv", parse_dates=["date"], usecols=cols)
    return df


def _load_emitter_map() -> dict[str, int]:
    """MOEX emitter_id для каждого ISIN — для issuer-level aggregation.

    Returns dict isin -> emitter_id.
    """
    u = pd.read_csv(DATA_DIR / "universe.csv", usecols=["isin", "emitter_id"])
    return dict(zip(u["isin"], u["emitter_id"]))


def compute_issuer_consensus(
    df: pd.DataFrame,
    z_col: str = "mispricing_z",
    blend_weight: float = 0.4,
) -> pd.DataFrame:
    """Add issuer-level consensus signal.

    Идея (Helwege-Huang-Wang JFI 2014, Bao-Hou JFE 2017):
    Если у эмитента несколько bonds, и **все** в Q5 по mispricing — strong consensus.
    Если 1 bond в Q5, остальные в Q3 — слабый, idiosyncratic.

    issuer_consensus_z = mean(z_col) over all bonds of same issuer on same date
    final_z = (1 - blend_weight) * z_col + blend_weight * issuer_consensus_z

    Args:
        df: signal df.
        z_col: bond-level z-score column.
        blend_weight: 0.0 = pure bond signal, 1.0 = pure issuer signal.

    Adds columns: emitter_id, issuer_consensus_z, blended_z.
    """
    df = df.copy()
    emitter_map = _load_emitter_map()
    df["emitter_id"] = df["isin"].map(emitter_map)

    # Issuer-level mean z per (date, emitter_id)
    df["issuer_consensus_z"] = (
        df.groupby(["date", "emitter_id"])[z_col].transform("mean")
    )
    df["issuer_n_bonds"] = (
        df.groupby(["date", "emitter_id"])["isin"].transform("nunique")
    )
    df["blended_z"] = (1 - blend_weight) * df[z_col] + blend_weight * df["issuer_consensus_z"]
    return df


def _load_trades_subset() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_DIR / "trades_daily.csv",
        parse_dates=["date"],
        usecols=["date", "isin", "close_price"],
    )
    return df


def _compute_cs_z_and_quintile(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Cross-sectional z-score + quintile (1=lowest, 5=highest) per date."""
    grp = df.groupby("date")[col]
    mu = grp.transform("mean")
    sd = grp.transform("std")
    df[f"{col}_z"] = (df[col] - mu) / sd.replace(0, np.nan)

    def _q(s: pd.Series) -> pd.Series:
        valid = s.notna()
        if valid.sum() < 5:
            return pd.Series([np.nan] * len(s), index=s.index)
        return pd.qcut(s.rank(method="first"), 5, labels=False) + 1

    df[f"{col}_quintile"] = grp.transform(_q)
    return df


def load_signal_full(
    predictions: ModelKind = "no_lag",
    include_roll: bool = True,
) -> pd.DataFrame:
    """Загрузить полностью обогащённый signal dataframe.

    Колонки результата:
        date, isin, g_spread_actual, g_spread_predicted, mispricing,
        mispricing_z, mispricing_quintile (1..5),
        close_price, price_return (clipped), coupon_daily, total_return,
        duration, rating_numeric, sector, volume_ma20, coupon_rate,
        coupon_frequency, time_to_maturity, issuer_name,
        net_debt_ebitda, debt_to_equity, roa, roe, rsbu_icr, ebitda_margin,
        tradable (bool), roll_spread_used (если include_roll).
    """
    preds = _load_predictions(predictions)
    panel = _load_panel_subset()
    trades = _load_trades_subset()

    df = preds.merge(panel, on=["date", "isin"], how="left")
    df = df.merge(trades, on=["date", "isin"], how="left")
    # Защита от случайных duplicate (date, isin) при upstream merge fan-out
    df = df.drop_duplicates(subset=["date", "isin"], keep="first")
    df = df.sort_values(["isin", "date"]).reset_index(drop=True)
    # NaN sector → 'Unknown', чтобы sector-neutral не дропал bonds silently
    df["sector"] = df["sector"].fillna("Unknown").astype(str)

    # Total return.
    # не обнуляем экстремальные returns в NaN — это маскировало
    # реальные price drops у stale bonds. Используем clip [-20%, +20%] чтобы capping
    # data-errors, но не давая «бесплатный купон» при просадке.
    df["close_prev"] = df.groupby("isin")["close_price"].shift(1)
    df["price_return"] = (df["close_price"] / df["close_prev"] - 1).clip(lower=-0.20, upper=0.20)
    df["coupon_daily"] = df["coupon_rate"].fillna(0) / 100 / 365
    df["total_return"] = df["price_return"].fillna(0) + df["coupon_daily"]

    df["tradable"] = df["volume_ma20"].fillna(0) > LIQUIDITY_MIN_RUB

    df = _compute_cs_z_and_quintile(df, "mispricing")

    # Roll spread (берём из signal.csv если он есть, иначе fallback)
    if include_roll:
        signal_path = DATA_DIR / "signal.csv"
        if signal_path.exists():
            roll = pd.read_csv(
                signal_path, parse_dates=["date"],
                usecols=["date", "isin", "roll_spread_used"],
            )
            df = df.merge(roll, on=["date", "isin"], how="left")
            df["roll_spread_used"] = (
                df["roll_spread_used"].fillna(ROLL_FALLBACK_BPS)
                .clip(lower=ROLL_MIN_BPS, upper=ROLL_MAX_BPS)
            )
        else:
            df["roll_spread_used"] = ROLL_FALLBACK_BPS

    return df


# Quality score (Houweling-vZ + AFP QMJ + Correia-Richardson-Tuna)

# Конфиг каждого компонента: (sign, default_weight, winsorize_range)
# sign = +1 если "higher is better quality", -1 если inverse.
QUALITY_COMPONENTS_DEFAULT: dict[str, tuple[int, float, tuple[float, float] | None]] = {
    "rating_numeric":   (+1, 1.5, None),         # high coverage anchor
    "roa":              (+1, 1.0, (-50, 50)),
    "roe":              (+1, 1.0, (-200, 200)),
    "debt_to_equity":   (-1, 1.0, (0, 5)),
    "rsbu_icr":         (+1, 0.8, (0, 20)),
    "net_debt_ebitda":  (-1, 0.8, (-5, 15)),
    "ebitda_margin":    (+1, 0.8, (-50, 80)),
}


def compute_quality_score(
    df: pd.DataFrame,
    components: dict[str, tuple[int, float, tuple[float, float] | None]] | None = None,
    min_components: int = 2,
) -> pd.DataFrame:
    """Cross-sectional quality composite z-score.

    Per (date, isin): z-score каждого компонента по cross-section этой даты,
    взвешенное среднее доступных компонентов (нормализация по сумме весов).

    Skip bond if available components < min_components.

    Adds column `quality_z` to df (NaN если компонентов недостаточно).
    """
    df = df.copy()
    components = components or QUALITY_COMPONENTS_DEFAULT

    # Per-component cross-sectional z-scores, with optional winsorization
    z_cols: list[str] = []
    weights: list[float] = []
    signs: list[int] = []

    for col, (sign, weight, wins) in components.items():
        if col not in df.columns:
            continue
        vals = df[col].copy()
        if wins is not None:
            lo, hi = wins
            vals = vals.clip(lower=lo, upper=hi)
        mu = vals.groupby(df["date"]).transform("mean")
        sd = vals.groupby(df["date"]).transform("std")
        z_col = f"_q_{col}_z"
        df[z_col] = sign * (vals - mu) / sd.replace(0, np.nan)
        z_cols.append(z_col)
        weights.append(weight)
        signs.append(sign)

    if not z_cols:
        df["quality_z"] = np.nan
        return df

    # Weighted average across available components
    z_arr = df[z_cols].to_numpy()  # shape (N, K)
    w_arr = np.array(weights)      # shape (K,)
    valid = ~np.isnan(z_arr)
    n_valid = valid.sum(axis=1)

    # Sum of weights for available components only
    w_matrix = np.where(valid, w_arr[None, :], 0.0)
    w_sum = w_matrix.sum(axis=1)
    z_filled = np.where(valid, z_arr, 0.0)
    weighted_sum = (z_filled * w_arr[None, :]).sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        composite = np.where(w_sum > 0, weighted_sum / np.where(w_sum > 0, w_sum, 1.0), np.nan)
    composite = np.where(n_valid < min_components, np.nan, composite)
    df["quality_z"] = composite
    df["quality_n_components"] = n_valid

    df = df.drop(columns=z_cols)
    return df


# Momentum signal (AMP 2013, Jostova et al 2013)

def compute_momentum(
    df: pd.DataFrame,
    lookback_days: int = 30,
    max_ffill: int = 10,
) -> pd.DataFrame:
    """Past-N-day cumulative return + cross-sectional z-score (per date).

    Calendar-aware: pivot к (date × isin), shift по date-index, поэтому lookback_days
    = N **уникальных торговых дат** в universe, а не N строк per bond. Допускается ffill
    цен максимум на max_ffill дней (на случай редких пропусков в данных одного bond).

    Adds columns `mom_return_<L>d` and `mom_z`.
    """
    L = lookback_days
    df = df.copy().sort_values(["isin", "date"]).reset_index(drop=True)

    # Drop existing momentum columns from previous calls to avoid merge collisions
    existing_mom = [c for c in df.columns if c.startswith("mom_return_") or c == "mom_z"]
    if existing_mom:
        df = df.drop(columns=existing_mom)

    # Pivot к (date × isin), shift на L unique trading dates
    price_wide = df.pivot_table(index="date", columns="isin", values="close_price")
    price_wide = price_wide.sort_index()
    # ffill ограниченно — чтобы не таскать stale prices через долгие пропуски
    price_ff = price_wide.ffill(limit=max_ffill)
    log_p = np.log(price_ff)
    mom_wide = np.exp(log_p - log_p.shift(L)) - 1

    col = f"mom_return_{L}d"
    mom_long = mom_wide.stack(future_stack=True).reset_index(name=col)
    df = df.merge(mom_long, on=["date", "isin"], how="left")

    df[col] = df[col].clip(lower=-0.50, upper=0.50)

    mu = df.groupby("date")[col].transform("mean")
    sd = df.groupby("date")[col].transform("std")
    df["mom_z"] = (df[col] - mu) / sd.replace(0, np.nan)

    return df


RebalanceFreq = Literal["M", "Q"]


def _rebalance_dates(dates: pd.Series, freq: RebalanceFreq) -> list[pd.Timestamp]:
    """Last trading day of each period (M or Q)."""
    if freq == "M":
        period = dates.dt.to_period("M")
    elif freq == "Q":
        period = dates.dt.to_period("Q")
    else:
        raise ValueError(f"Unsupported freq: {freq}")
    last_per_period = dates.groupby(period).max()
    return sorted(last_per_period.tolist())


PositionSizing = Literal["equal", "z_weighted", "risk_parity"]


def _compute_weights(
    selected: pd.DataFrame,
    sizing: PositionSizing,
    ranking_col: str,
    vol_lookback: int = 60,
    vol_data: pd.DataFrame | None = None,
) -> np.ndarray:
    """Веса позиций. Возвращает array той же длины, сумма = 1.

    - 'equal': 1/N для каждой
    - 'z_weighted': пропорционально (ranking_col - min(ranking_col) + ε), нормированы
    - 'risk_parity': 1/σ_60d (требует vol_data с колонкой 'realized_vol')
    """
    n = len(selected)
    if n == 0:
        return np.array([])
    if sizing == "equal":
        return np.full(n, 1.0 / n)
    if sizing == "z_weighted":
        vals = selected[ranking_col].to_numpy()
        # guard от NaN-poisoning. Если в vals есть NaN — fall back к equal.
        if np.any(~np.isfinite(vals)):
            return np.full(n, 1.0 / n)
        shifted = vals - vals.min() + 1e-6
        s = shifted.sum()
        if s <= 0:
            return np.full(n, 1.0 / n)
        w = shifted / s
        return w
    if sizing == "risk_parity":
        if vol_data is None or "realized_vol" not in selected.columns:
            return np.full(n, 1.0 / n)
        vols = selected["realized_vol"].to_numpy()
        # Заменим NaN/0 на медианную волу
        med_vol = np.nanmedian(vols) if np.any(np.isfinite(vols) & (vols > 0)) else 0.01
        vols = np.where(np.isfinite(vols) & (vols > 0), vols, med_vol)
        inv = 1.0 / vols
        return inv / inv.sum()
    raise ValueError(f"Unknown sizing: {sizing}")


def _select_at_rebalance(
    day_df: pd.DataFrame,
    n_max: int,
    ranking_col: str,
    sector_neutral: bool,
    n_per_sector: int,
) -> pd.DataFrame:
    """Selection logic on one rebalance date.

    Если sector_neutral=True: top n_per_sector в каждом секторе, потом cap n_max.
    Иначе: top n_max по ranking_col globally.
    """
    day_df = day_df.dropna(subset=[ranking_col])
    if day_df.empty:
        return day_df
    if sector_neutral:
        # Top n_per_sector в каждом секторе.
        # NB: groupby(...).apply дропает grouping column в new pandas →
        # включаем sector через include_groups для совместимости, или восстанавливаем.
        per_sec_list = []
        for sec_name, g in day_df.groupby("sector", group_keys=False):
            top = g.nlargest(n_per_sector, ranking_col).copy()
            top["sector"] = sec_name  # ensure column preserved
            per_sec_list.append(top)
        if not per_sec_list:
            return day_df.iloc[0:0]
        per_sec = pd.concat(per_sec_list, ignore_index=True)
        # Если суммарно > n_max, оставляем top n_max по ranking_col
        if len(per_sec) > n_max:
            per_sec = per_sec.nlargest(n_max, ranking_col)
        return per_sec.reset_index(drop=True)
    return day_df.nlargest(n_max, ranking_col).reset_index(drop=True)


def backtest_rule_c(
    sig: pd.DataFrame,
    n_max: int = N_MAX_DEFAULT,
    freq: RebalanceFreq = "M",
    ranking_col: str = "mispricing",
    filter_mask: pd.Series | None = None,
    require_tradable: bool = True,
    cost_override: float | None = None,
    sector_neutral: bool = False,
    n_per_sector: int = 2,
    sizing: PositionSizing = "equal",
    vol_lookback: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rule C backtest: rebalance каждый период по top-N по ranking_col.

    Args:
        sig: должен иметь date, isin, total_return, close_price, roll_spread_used,
             tradable, plus ranking_col.
        n_max: макс позиций.
        freq: 'M' monthly, 'Q' quarterly.
        ranking_col: столбец для ранжирования (descending = больше = лучше).
        filter_mask: bool Series (длина = len(sig)), True = bond eligible.
                     Если None, eligibility только по tradable.
        require_tradable: применять `sig['tradable']`.
        cost_override: если не None, используем этот round-trip cost вместо roll_spread_used.
        sector_neutral: если True, берём top-n_per_sector в каждом секторе вместо глобального top-N.
        n_per_sector: сколько bonds в каждом секторе (только при sector_neutral=True).
        sizing: 'equal' (default), 'z_weighted' (вес ∝ ranking), 'risk_parity' (вес ∝ 1/σ).

    Returns:
        (daily_returns, trades)
        daily_returns: DataFrame[date, return, n_positions]
        trades:        DataFrame[date, isin, side, cost]
    """
    sig = sig.copy()
    if filter_mask is not None:
        if len(filter_mask) != len(sig):
            raise ValueError("filter_mask length must match sig length")
        # Index-safe alignment: reindex mask на sig.index если индекс mask отличается.
        if isinstance(filter_mask, pd.Series) and not filter_mask.index.equals(sig.index):
            raise ValueError(
                "filter_mask.index must equal sig.index — pass mask без reset_index/sort"
            )
        sig["_eligible"] = filter_mask.to_numpy().astype(bool)
    else:
        sig["_eligible"] = True
    if require_tradable:
        sig["_eligible"] = sig["_eligible"] & sig["tradable"].astype(bool)

    # Для risk_parity вычисляем rolling realized vol per bond (один раз)
    if sizing == "risk_parity" and "realized_vol" not in sig.columns:
        sig = sig.sort_values(["isin", "date"]).copy()
        sig["realized_vol"] = (
            sig.groupby("isin")["total_return"]
            .transform(lambda s: s.rolling(vol_lookback, min_periods=vol_lookback // 2).std())
        )

    rebal_dates = _rebalance_dates(sig["date"], freq=freq)

    # Build selections (с поддержкой sector-neutral)
    selections = []
    for rdate in rebal_dates:
        day_df = sig[(sig["date"] == rdate) & sig["_eligible"]]
        if day_df.empty:
            selections.append({"rebalance_date": rdate, "isins": [],
                               "roll_spreads": {}, "weights": {}})
            continue
        eligible = _select_at_rebalance(
            day_df, n_max=n_max, ranking_col=ranking_col,
            sector_neutral=sector_neutral, n_per_sector=n_per_sector,
        )
        if eligible.empty:
            selections.append({"rebalance_date": rdate, "isins": [],
                               "roll_spreads": {}, "weights": {}})
            continue
        if cost_override is not None:
            roll = {i: cost_override for i in eligible["isin"]}
        else:
            roll = dict(zip(eligible["isin"], eligible["roll_spread_used"]))
        weights = _compute_weights(eligible, sizing=sizing, ranking_col=ranking_col)
        weight_map = dict(zip(eligible["isin"], weights))
        selections.append({"rebalance_date": rdate, "isins": eligible["isin"].tolist(),
                           "roll_spreads": roll, "weights": weight_map})
    sels = pd.DataFrame(selections).sort_values("rebalance_date").reset_index(drop=True)

    sig_idx = sig.set_index(["date", "isin"])
    all_dates = np.sort(sig["date"].unique())

    daily_rows: list[dict] = []
    trade_rows: list[dict] = []

    for idx in range(len(sels) - 1):
        start = sels.iloc[idx]["rebalance_date"]
        end = sels.iloc[idx + 1]["rebalance_date"]
        isins = sels.iloc[idx]["isins"]
        roll_spreads = sels.iloc[idx]["roll_spreads"]
        weights = sels.iloc[idx]["weights"]
        if not isins:
            continue
        period_dates = [d for d in all_dates if start < d <= end]
        if not period_dates:
            continue
        n = len(isins)
        # weighted average cost (для unification: cost per RUB invested)
        avg_cost = float(sum(weights.get(i, 0.0) * roll_spreads.get(i, ROLL_FALLBACK_BPS) / 2
                              for i in isins))

        for isin in isins:
            cost_leg = roll_spreads.get(isin, ROLL_FALLBACK_BPS) / 2
            trade_rows.append({"date": start, "isin": isin, "side": "BUY",
                               "cost": cost_leg, "weight": weights.get(isin, 0.0)})
            trade_rows.append({"date": end, "isin": isin, "side": "SELL",
                               "cost": cost_leg, "weight": weights.get(isin, 0.0)})

        # Daily returns с position weights. Entry cost списывается на period_dates[0],
        # exit на [-1] — но ТОЛЬКО если на этот день есть хотя бы одна live position
        # (иначе phantom cost без offset return).
        for d in period_dates:
            r_port = 0.0
            w_total = 0.0
            for isin in isins:
                key = (d, isin)
                w_i = weights.get(isin, 0.0)
                if key in sig_idx.index:
                    r_i = sig_idx.loc[key, "total_return"]
                    if pd.notna(r_i):
                        r_port += w_i * float(r_i)
                        w_total += w_i
            if w_total > 0:
                r_port = r_port / w_total
                # Cost applied только при live positions
                if d == period_dates[0]:
                    r_port -= avg_cost
                if d == period_dates[-1]:
                    r_port -= avg_cost
            daily_rows.append({"date": d, "return": r_port, "n_positions": n})

    return pd.DataFrame(daily_rows), pd.DataFrame(trade_rows)


def calc_metrics(rets: pd.DataFrame, capital: float = DEFAULT_CAPITAL,
                 name: str = "") -> dict:
    """Standard performance metrics from daily returns DataFrame."""
    if len(rets) == 0:
        return {"name": name, "n_days": 0}
    r = rets["return"].to_numpy()
    eq = np.cumprod(1.0 + r)
    n_days = len(r)
    # CAGR — calendar-elapsed years (не trading-days/252), чтобы пропущенные дни
    # не завышали ставку компаундирования.
    if "date" in rets.columns and len(rets) >= 2:
        d_min = pd.to_datetime(rets["date"].min())
        d_max = pd.to_datetime(rets["date"].max())
        n_years_cal = max((d_max - d_min).days / 365.25, 1.0 / 365.25)
    else:
        n_years_cal = n_days / 252.0
    n_years = n_days / 252.0  # trading-year used for Sharpe annualization (correct convention)
    total_ret = float(eq[-1] - 1.0)
    cagr = float(eq[-1] ** (1.0 / n_years_cal) - 1.0) if n_years_cal > 0 else np.nan
    std = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    sharpe = float(np.sqrt(252) * r.mean() / std) if std > 0 else np.nan
    # Sortino — canonical (Sortino 1991): downside deviation = √(mean(min(r,0)²)),
    # МAR = 0.
    neg = np.minimum(r, 0.0)
    dd_dev = float(np.sqrt((neg ** 2).mean())) if len(neg) > 0 else 0.0
    sortino = float(np.sqrt(252) * r.mean() / dd_dev) if dd_dev > 0 else np.nan
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = float(dd.min()) if len(dd) > 0 else np.nan
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else np.nan
    hit_rate = float((r > 0).mean())
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    pf = float(gains / losses) if losses > 0 else np.nan
    avg_n = float(rets["n_positions"].mean()) if "n_positions" in rets.columns else np.nan
    end_cap = capital * float(eq[-1])
    profit = end_cap - capital

    return {
        "name": name,
        "n_days": n_days,
        "n_years_calendar": round(n_years_cal, 2),
        "n_years_trading": round(n_years, 2),
        "total_return": total_ret,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": max_dd,
        "Calmar": calmar,
        "Hit rate": hit_rate,
        "Profit factor": pf,
        "avg_n_positions": avg_n,
        "end_capital_RUB": end_cap,
        "profit_RUB": profit,
        "profit_per_year_RUB": profit / max(n_years_cal, 0.001),
    }


def split_main_holdout(rets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split daily returns into main period (≤ 2025-09-30) and clean holdout (≥ 2025-10-01)."""
    main = rets[rets["date"] <= MAIN_END].copy()
    holdout = rets[rets["date"] >= HOLDOUT_START].copy()
    return main, holdout


def block_bootstrap_sharpe(
    daily_ret: np.ndarray,
    n_boot: int = 1000,
    block: int = 60,
    seed: int = 42,
) -> np.ndarray:
    """Block bootstrap distribution of annualized Sharpe ratio.

    Args:
        daily_ret: 1D array of daily returns.
        n_boot: number of bootstrap iterations.
        block: block length in trading days (default 60 = ~3 months).
        seed: RNG seed.

    Returns:
        1D ndarray of bootstrap Sharpes (length n_boot).
    """
    daily_ret = np.asarray(daily_ret, dtype=float)
    n = len(daily_ret)
    if n == 0:
        return np.full(n_boot, np.nan)
    # Если series слишком короткая для содержательного MBB (один блок) — возвращаем NaN.
    if n < 2 * block:
        return np.full(n_boot, np.nan)
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
        sample = np.concatenate([daily_ret[s:s + block] for s in starts])[:n]
        sd = sample.std(ddof=1)
        out[b] = np.sqrt(252) * sample.mean() / sd if sd > 0 else np.nan
    return out


def compare_variants(
    variants: dict[str, pd.DataFrame],
    capital: float = DEFAULT_CAPITAL,
) -> pd.DataFrame:
    """Build side-by-side metrics table for multiple variants."""
    rows = [calc_metrics(rets, capital=capital, name=name)
            for name, rets in variants.items()]
    return pd.DataFrame(rows).set_index("name")


def load_benchmarks() -> pd.DataFrame:
    """Load benchmark indices (RUCBTRNS, RGBITR) с daily returns.

    Returns DataFrame[date, RUCBTRNS, RGBITR, rucbtrns_ret, rgbitr_ret].
    """
    path = DATA_DIR / "benchmarks.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def _align_returns(
    strategy_rets: pd.DataFrame,
    benchmark_rets: pd.DataFrame,
    bench_col: str = "rucbtrns_ret",
) -> pd.DataFrame:
    """Merge strategy daily returns с benchmark на inner-join by date.

    Args:
        strategy_rets: DataFrame[date, return, ...]
        benchmark_rets: DataFrame[date, ..., bench_col]
        bench_col: name of benchmark return column.

    Returns DataFrame[date, r_strat, r_bench] (inner join).
    """
    s = strategy_rets[["date", "return"]].rename(columns={"return": "r_strat"})
    b = benchmark_rets[["date", bench_col]].rename(columns={bench_col: "r_bench"})
    aligned = s.merge(b, on="date", how="inner").dropna(subset=["r_strat", "r_bench"])
    return aligned.reset_index(drop=True)


def alpha_regression(
    strategy_rets: pd.DataFrame,
    benchmark_rets: pd.DataFrame,
    bench_col: str = "rucbtrns_ret",
    nw_lags: int = 12,
) -> dict:
    """OLS r_strat = α + β·r_bench + ε с Newey-West HAC standard errors.

    Возвращает annualized α (умножен на 252), t-stat, p-value, β, R², n.
    Литература: Newey & West (1987), Andrews (1991). nw_lags=12 рекомендация Andrews.
    """
    import statsmodels.api as sm

    aligned = _align_returns(strategy_rets, benchmark_rets, bench_col)
    if len(aligned) < 30:
        return {"alpha_annual": np.nan, "alpha_t_stat": np.nan, "alpha_p_value": np.nan,
                "beta": np.nan, "r_squared": np.nan, "n_obs": len(aligned)}

    y = aligned["r_strat"].to_numpy()
    X = sm.add_constant(aligned["r_bench"].to_numpy())
    # OLS с Newey-West HAC SE
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})

    alpha_daily = float(model.params[0])
    beta = float(model.params[1])
    t_alpha = float(model.tvalues[0])
    p_alpha = float(model.pvalues[0])

    return {
        "alpha_daily": alpha_daily,
        "alpha_annual": alpha_daily * 252,
        "alpha_t_stat": t_alpha,
        "alpha_p_value": p_alpha,
        "beta": beta,
        "r_squared": float(model.rsquared),
        "n_obs": len(aligned),
    }


def information_ratio(
    strategy_rets: pd.DataFrame,
    benchmark_rets: pd.DataFrame,
    bench_col: str = "rucbtrns_ret",
) -> dict:
    """Information Ratio = annualized mean(excess) / annualized std(excess).

    excess = r_strat - r_bench.
    """
    aligned = _align_returns(strategy_rets, benchmark_rets, bench_col)
    if len(aligned) < 2:
        return {"IR": np.nan, "mean_excess_ann": np.nan,
                "tracking_error_ann": np.nan, "n_obs": len(aligned)}
    excess = (aligned["r_strat"] - aligned["r_bench"]).to_numpy()
    mu = float(excess.mean()) * 252
    sd = float(excess.std(ddof=1)) * np.sqrt(252)
    ir = mu / sd if sd > 0 else np.nan
    return {"IR": ir, "mean_excess_ann": mu, "tracking_error_ann": sd, "n_obs": len(aligned)}


def sharpe_difference_test(
    strategy_rets: pd.DataFrame,
    benchmark_rets: pd.DataFrame,
    bench_col: str = "rucbtrns_ret",
    n_boot: int = 1000,
    block: int = 60,
    seed: int = 42,
) -> dict:
    """Bootstrap-based test для H0: Sharpe_strategy <= Sharpe_benchmark.

    Stationary block bootstrap (Politis & Romano 1994) разности дневных returns.
    P-value одностороннее (Sharpe_strategy > Sharpe_benchmark).
    """
    aligned = _align_returns(strategy_rets, benchmark_rets, bench_col)
    n = len(aligned)
    if n < 2 * block:
        return {"sharpe_strat": np.nan, "sharpe_bench": np.nan, "diff": np.nan,
                "p_value": np.nan, "n_obs": n}

    rs = aligned["r_strat"].to_numpy()
    rb = aligned["r_bench"].to_numpy()
    sd_s = rs.std(ddof=1)
    sd_b = rb.std(ddof=1)
    sh_s = np.sqrt(252) * rs.mean() / sd_s if sd_s > 0 else np.nan
    sh_b = np.sqrt(252) * rb.mean() / sd_b if sd_b > 0 else np.nan
    diff = sh_s - sh_b

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        rs_b = rs[idx]
        rb_b = rb[idx]
        sd_sb = rs_b.std(ddof=1)
        sd_bb = rb_b.std(ddof=1)
        sh_sb = np.sqrt(252) * rs_b.mean() / sd_sb if sd_sb > 0 else 0.0
        sh_bb = np.sqrt(252) * rb_b.mean() / sd_bb if sd_bb > 0 else 0.0
        diffs[b] = sh_sb - sh_bb

    # One-sided p-value: P(diff_bootstrap <= 0) под null'ом «без edge»
    # Сначала центрируем relative к observed mean diff
    centered = diffs - diffs.mean()
    p_value = float((centered >= diff).mean())
    return {"sharpe_strat": float(sh_s), "sharpe_bench": float(sh_b),
            "diff": float(diff), "p_value": p_value, "n_obs": n,
            "bootstrap_ci": (float(np.percentile(diffs, 2.5)),
                             float(np.percentile(diffs, 97.5)))}


def compare_to_benchmark(
    strategy_rets: pd.DataFrame,
    benchmark_rets: pd.DataFrame,
    bench_col: str = "rucbtrns_ret",
    name: str = "",
) -> dict:
    """Один-shot full comparison: alpha regression + IR + Sharpe test + raw metrics."""
    reg = alpha_regression(strategy_rets, benchmark_rets, bench_col=bench_col)
    ir = information_ratio(strategy_rets, benchmark_rets, bench_col=bench_col)
    sh_test = sharpe_difference_test(strategy_rets, benchmark_rets, bench_col=bench_col)
    return {
        "name": name,
        "n_obs": reg["n_obs"],
        "alpha_annual": reg["alpha_annual"],
        "alpha_t_stat": reg["alpha_t_stat"],
        "alpha_p_value": reg["alpha_p_value"],
        "beta": reg["beta"],
        "r_squared": reg["r_squared"],
        "IR": ir["IR"],
        "mean_excess_ann": ir["mean_excess_ann"],
        "tracking_error_ann": ir["tracking_error_ann"],
        "sharpe_strat": sh_test["sharpe_strat"],
        "sharpe_bench": sh_test["sharpe_bench"],
        "sharpe_diff": sh_test["diff"],
        "sharpe_diff_p_value": sh_test["p_value"],
    }


# Hierarchical Risk Parity (De Prado 2016, JPM)

def _correlation_distance(corr: np.ndarray) -> np.ndarray:
    """Distance metric: d_ij = sqrt(0.5 * (1 - corr_ij))."""
    d = np.sqrt(0.5 * (1.0 - np.clip(corr, -1.0, 1.0)))
    np.fill_diagonal(d, 0.0)
    return d


def _quasi_diagonal_order(linkage_matrix: np.ndarray, n_leaves: int) -> list[int]:
    """Recover quasi-diagonal order of leaves from scipy linkage matrix."""
    # linkage_matrix shape (n-1, 4); ids 0..n-1 are leaves, n..2n-2 internal
    order: list[int] = []

    def _recurse(node_id: int) -> None:
        if node_id < n_leaves:
            order.append(int(node_id))
            return
        row = linkage_matrix[int(node_id) - n_leaves]
        left, right = int(row[0]), int(row[1])
        _recurse(left)
        _recurse(right)

    root = 2 * n_leaves - 2
    _recurse(root)
    return order


def _inverse_variance_weight(cov: np.ndarray, indices: list[int]) -> np.ndarray:
    """Inverse-variance weights for cluster, normalized to sum=1.

    Guard против zero-variance factors (B23 fix).
    """
    sub_cov = cov[np.ix_(indices, indices)]
    variances = np.diag(sub_cov)
    # Replace zero/negative variances с small positive epsilon (защита от 1/0 → inf)
    variances = np.where(variances > 1e-12, variances, 1e-12)
    ivp = 1.0 / variances
    return ivp / ivp.sum()


def hrp_weights(returns: pd.DataFrame, eps: float = 1e-8) -> pd.Series:
    """Hierarchical Risk Parity weights (De Prado 2016).

    Args:
        returns: DataFrame[date, factors] with daily returns of N factor portfolios.

    Returns:
        Series indexed by factor name, weights sum to 1.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    R = returns.dropna(how="all").copy()
    if R.shape[1] < 2:
        # Single factor → 100%
        cols = R.columns.tolist()
        return pd.Series([1.0] * len(cols), index=cols)

    cov = R.cov().to_numpy()
    corr = R.corr().to_numpy()
    # Replace NaN corr with 0 (uncorrelated)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)

    dist = _correlation_distance(corr)
    # Hierarchical clustering via single linkage on condensed distance
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")
    order = _quasi_diagonal_order(link, n_leaves=cov.shape[0])

    # Recursive bisection
    weights = np.ones(cov.shape[0])
    clusters = [order[:]]
    while clusters:
        new_clusters: list[list[int]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]
            # Inverse-variance weight per side, take cluster variance
            w_left = _inverse_variance_weight(cov, left)
            v_left = float(w_left @ cov[np.ix_(left, left)] @ w_left)
            w_right = _inverse_variance_weight(cov, right)
            v_right = float(w_right @ cov[np.ix_(right, right)] @ w_right)
            alpha = 1.0 - v_left / max(v_left + v_right, eps)  # weight for LEFT
            for idx in left:
                weights[idx] *= alpha
            for idx in right:
                weights[idx] *= (1.0 - alpha)
            new_clusters.extend([left, right])
        clusters = new_clusters

    weights = weights / weights.sum()
    return pd.Series(weights, index=R.columns)


def rolling_hrp_allocation(
    factor_returns: pd.DataFrame,
    lookback: int = 252,
    rebalance_freq: str = "Q",
) -> pd.DataFrame:
    """Roll HRP allocation on factor returns.

    Args:
        factor_returns: DataFrame[date, factor1, factor2, ...] daily returns.
        lookback: history window for cov estimation (default 252 days).
        rebalance_freq: 'M' or 'Q'.

    Returns:
        DataFrame[date, factor1, ..., return]:
            * weights ffilled между rebalance датами
            * 'return' = sum of weighted factor returns daily
    """
    df = factor_returns.copy().sort_values("date").reset_index(drop=True)
    factor_cols = [c for c in df.columns if c != "date"]

    rebal_dates = _rebalance_dates(df["date"], freq=rebalance_freq)
    weights_records: list[dict] = []
    for rdate in rebal_dates:
        history = df[(df["date"] < rdate) & (df["date"] >= rdate - pd.Timedelta(days=int(lookback * 1.5)))]
        history = history.dropna(subset=factor_cols, how="any")
        if len(history) < lookback // 2:
            continue
        history = history.tail(lookback)
        w = hrp_weights(history[factor_cols])
        rec = {"date": rdate}
        rec.update(w.to_dict())
        weights_records.append(rec)

    if not weights_records:
        raise ValueError("Insufficient history for HRP rolling allocation")

    w_df = pd.DataFrame(weights_records).sort_values("date").reset_index(drop=True)

    # ffill weights to daily
    all_dates = df["date"].copy()
    w_daily = w_df.set_index("date").reindex(all_dates).ffill().reset_index()
    w_daily.columns = ["date"] + factor_cols

    merged = df.merge(w_daily, on="date", suffixes=("_ret", "_w"))
    ret_cols = [f"{c}_ret" for c in factor_cols]
    w_cols = [f"{c}_w" for c in factor_cols]
    ret_arr = merged[ret_cols].fillna(0).to_numpy()
    w_arr = merged[w_cols].fillna(0).to_numpy()
    merged["return"] = (ret_arr * w_arr).sum(axis=1)
    merged["n_positions"] = len(factor_cols)

    out = merged[["date", "return", "n_positions"] + w_cols].copy()
    out.columns = ["date", "return", "n_positions"] + [c.replace("_w", "_weight") for c in w_cols]
    return out


REGIMES: list[tuple[str, pd.Timestamp, pd.Timestamp]] = [
    ("A: pre-СВО",   pd.Timestamp("2021-01-04"), pd.Timestamp("2022-02-23")),
    ("B: шок 2022",  pd.Timestamp("2022-02-24"), pd.Timestamp("2022-12-31")),
    ("C: норма",     pd.Timestamp("2023-01-01"), MAIN_END),
]


def sub_period_metrics(
    rets: pd.DataFrame,
    variant_name: str = "",
    capital: float = DEFAULT_CAPITAL,
) -> pd.DataFrame:
    """Метрики по 3 режимам A/B/C на Main period."""
    rows = []
    for rname, start, end in REGIMES:
        sub = rets[(rets["date"] >= start) & (rets["date"] <= end)]
        m = calc_metrics(sub, capital=capital, name=f"{variant_name} | {rname}")
        rows.append(m)
    return pd.DataFrame(rows)
