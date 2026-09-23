"""
Build the final modeling panel dataset from raw data sources.

Merges trades, universe, issuer info, macro, and ratings into a single
panel with G-spread as the target variable.  Applies a liquidity filter
(configurable) and computes derived features.

Output: data/panel.csv

Usage:
    python scripts/build_panel.py                  # full build
    python scripts/build_panel.py --test           # first 3 months only
    python scripts/build_panel.py --no-liquidity-filter  # skip liquidity filter
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUTPUT_FILE = DATA_DIR / "panel.csv"

# Start date: 2020-01-01 — before this, fundamental data is missing for most issuers.
# Also mitigates survivorship bias (too few liquid bonds in 2019).
PANEL_START = "2020-01-01"

TRADES_FILE = DATA_DIR / "trades_daily.csv"
OFZ_CURVE_FILE = DATA_DIR / "ofz_curve.csv"
MACRO_FILE = DATA_DIR / "macro.csv"
UNIVERSE_FILE = DATA_DIR / "universe.csv"
ISSUER_INFO_FILE = DATA_DIR / "issuer_info.csv"
RATINGS_FILE = DATA_DIR / "ratings.csv"
MULTIPLIERS_FILE = DATA_DIR / "multipliers.csv"
RSBU_FILE = DATA_DIR / "fundamentals_rsbu.csv"

TEST_MODE_MONTHS = 3

MIN_ACTIVE_DAYS_PER_MONTH = 15
MIN_AVG_VOLUME_RUB = 1_000_000

# Rating agencies in priority order (highest priority first)
RATING_AGENCIES = ["АКРА", "Эксперт РА", "НКР", "НРА"]

# National scale patterns per agency (to filter out ESG / self-assessment)
NATIONAL_SCALE_PATTERNS: dict[str, str] = {
    "АКРА": "Национальная рейтинговая шкала АКРА",
    "Эксперт РА": "Национальная российская рейтинговая шкала",
    "НКР": "Национальная рейтинговая шкала для Российской Федерации",
    "НРА": "Национальная кредитная рейтинговая шкала",
}

# Standard rating scale: AAA=1 .. D=20
RATING_SCALE: dict[str, int] = {
    "AAA": 1,
    "AA+": 2,
    "AA": 3,
    "AA-": 4,
    "A+": 5,
    "A": 6,
    "A-": 7,
    "BBB+": 8,
    "BBB": 9,
    "BBB-": 10,
    "BB+": 11,
    "BB": 12,
    "BB-": 13,
    "B+": 14,
    "B": 15,
    "B-": 16,
    "CCC+": 17,
    "CCC": 18,
    "CCC-": 19,
    "D": 20,
}

# Known state-owned companies (substrings to match against issuer_name)
STATE_COMPANIES = [
    "Роснефть",
    "Газпром",
    "Сбер",
    "ВТБ",
    "РЖД",
    "Ростелеком",
    "Россети",
    "Транснефть",
    "Алроса",
    "Аэрофлот",
    "Дом.РФ",
    "ГТЛК",
    "РусГидро",
    "Совкомфлот",
    "Росатом",
    "Атомэнергопром",
    "ФСК",
]

OUTPUT_COLUMNS = [
    "date",
    "isin",
    "g_spread",
    # Tier 1 features
    "duration",
    "rating_numeric",
    "net_debt_ebitda",
    "ebitda_margin",
    "ebitda_yoy",
    "revenue_yoy",
    "debt_to_equity",
    "roe",
    "roa",
    "rsbu_net_debt",
    "rsbu_ebit",
    "rsbu_interest_paid",
    "rsbu_icr",
    "rsbu_total_assets",
    "rsbu_revenue",
    "key_rate",
    "ofz_slope",
    "rvi",
    "oil_brent",
    "usdrub",
    "sector",
    "issue_volume_rub",
    # Tier 2 features
    "imoex_return",
    "volume_ma20",
    "coupon_rate",
    "age_days",
    "imoex_realized_vol",
    "time_to_maturity",
    "coupon_frequency",
    "is_state_owned",
    "issuer_name",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Regex to strip agency-specific suffixes: (RU), |ru|, .ru, ru-prefix
# Examples: BBB-(RU) -> BBB-, ruBBB- -> BBB-, BBB-.ru -> BBB-, BB|ru| -> BB
_RATING_SUFFIX_RE = re.compile(
    r"^(?:ru)?"         # optional "ru" prefix (Эксперт РА style)
    r"([A-Da-d][A-Da-d+\-]*?)"  # core rating (e.g. BBB-, A+, D)
    r"(?:\(RU\)|\|ru\||\.ru)?$",  # optional suffix
    re.IGNORECASE,
)


def normalize_rating(raw: str) -> str | None:
    """Normalize a Russian national-scale rating to a standard form (e.g. BBB-).

    Returns None for non-parseable or withdrawn ratings.
    """
    if not isinstance(raw, str):
        return None

    cleaned = raw.strip()
    if cleaned.lower() in ("withdrawn", ""):
        return None

    m = _RATING_SUFFIX_RE.match(cleaned)
    if m is None:
        return None

    core = m.group(1).upper()

    # Handle edge case: single letters like "A" or "B" or "D"
    # Ensure we have a valid rating
    if core not in RATING_SCALE:
        return None

    return core


def rating_to_numeric(rating_str: str | None) -> float:
    """Convert a normalized rating string to a numeric value.

    Returns NaN if rating is None or unrecognized.
    """
    if rating_str is None:
        return np.nan
    return RATING_SCALE.get(rating_str, np.nan)


def load_trades(test_mode: bool) -> pd.DataFrame:
    """Load daily trades and optionally trim to first N months."""
    logger.info("Loading trades from %s", TRADES_FILE)
    df = pd.read_csv(TRADES_FILE, parse_dates=["date"])

    if test_mode:
        cutoff = df["date"].min() + pd.DateOffset(months=TEST_MODE_MONTHS)
        df = df[df["date"] <= cutoff].copy()
        logger.info(
            "TEST MODE: trimmed trades to first %d months (%d rows, up to %s)",
            TEST_MODE_MONTHS,
            len(df),
            cutoff.strftime("%Y-%m-%d"),
        )

    # Convert duration from days to years
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce") / 365.0

    df["yield_close"] = pd.to_numeric(df["yield_close"], errors="coerce")
    df["volume_rub"] = pd.to_numeric(df["volume_rub"], errors="coerce")

    # Filter by PANEL_START to avoid sparse fundamental coverage in early years
    before = len(df)
    df = df[df["date"] >= pd.Timestamp(PANEL_START)].reset_index(drop=True)
    logger.info("Filtered by PANEL_START=%s: %d -> %d rows", PANEL_START, before, len(df))

    logger.info(
        "Trades loaded: %d rows, %d ISINs, date range %s .. %s",
        len(df),
        df["isin"].nunique(),
        df["date"].min().strftime("%Y-%m-%d"),
        df["date"].max().strftime("%Y-%m-%d"),
    )
    return df


def load_ofz_curve() -> pd.DataFrame:
    """Load OFZ zero-coupon curve data."""
    logger.info("Loading OFZ curve from %s", OFZ_CURVE_FILE)
    df = pd.read_csv(OFZ_CURVE_FILE, parse_dates=["date"])
    logger.info("OFZ curve loaded: %d rows, tenors: %s", len(df), sorted(df["tenor_years"].unique()))
    return df


def load_macro() -> pd.DataFrame:
    """Load macroeconomic indicators."""
    logger.info("Loading macro from %s", MACRO_FILE)
    df = pd.read_csv(MACRO_FILE, parse_dates=["date"])
    logger.info("Macro loaded: %d rows", len(df))
    return df


def load_universe() -> pd.DataFrame:
    """Load bond universe metadata."""
    logger.info("Loading universe from %s", UNIVERSE_FILE)
    df = pd.read_csv(
        UNIVERSE_FILE,
        usecols=[
            "isin",
            "issuer_name",
            "issue_date",
            "maturity_date",
            "coupon_rate",
            "coupon_frequency",
            "issue_volume_rub",
        ],
        parse_dates=["issue_date", "maturity_date"],
    )
    logger.info("Universe loaded: %d bonds", len(df))
    return df


def load_issuer_info() -> pd.DataFrame:
    """Load issuer information (sector, company name)."""
    logger.info("Loading issuer info from %s", ISSUER_INFO_FILE)
    df = pd.read_csv(ISSUER_INFO_FILE, usecols=["isin", "sector", "company_name", "company_id"])
    logger.info("Issuer info loaded: %d rows", len(df))
    return df


def load_ratings() -> pd.DataFrame:
    """Load and preprocess ratings: filter agencies, national scale, normalize.

    Returns a DataFrame with columns: isin, date, agency, rating_normalized, rating_numeric
    sorted by isin, date, agency_priority.
    """
    logger.info("Loading ratings from %s", RATINGS_FILE)
    df = pd.read_csv(RATINGS_FILE)

    df = df[df["agency"].isin(RATING_AGENCIES)].copy()
    logger.info("Ratings after agency filter: %d rows", len(df))

    # Filter to national scale only (per agency)
    mask = pd.Series(False, index=df.index)
    for agency, pattern in NATIONAL_SCALE_PATTERNS.items():
        agency_mask = (df["agency"] == agency) & (
            df["scale"].str.contains(pattern, na=False)
        )
        mask |= agency_mask
    df = df[mask].copy()
    logger.info("Ratings after national scale filter: %d rows", len(df))

    df["date"] = pd.to_datetime(df["date_assigned"], format="%d.%m.%Y", errors="coerce")
    df = df.dropna(subset=["date"])

    df["rating_normalized"] = df["rating"].apply(normalize_rating)
    df = df.dropna(subset=["rating_normalized"])
    logger.info("Ratings after normalization (excl. Withdrawn): %d rows", len(df))

    df["rating_numeric"] = df["rating_normalized"].apply(rating_to_numeric)

    # Assign agency priority (lower = higher priority)
    agency_priority = {a: i for i, a in enumerate(RATING_AGENCIES)}
    df["agency_priority"] = df["agency"].map(agency_priority)

    # Sort: for merge_asof we need sorted by date; within same isin+date
    # pick highest-priority agency
    df = df.sort_values(["isin", "date", "agency_priority"])
    df = df.drop_duplicates(subset=["isin", "date"], keep="first")

    df = df[["isin", "date", "rating_numeric"]].copy()
    logger.info("Ratings final (deduplicated): %d rows", len(df))
    return df


def compute_g_spread(trades: pd.DataFrame, ofz: pd.DataFrame) -> pd.DataFrame:
    """Compute G-spread = yield_close - interpolated OFZ yield at bond duration.

    Vectorized via groupby("date") + np.interp.
    """
    logger.info("Computing G-spread...")

    # Pre-sort OFZ curve by date and tenor for efficient lookup
    ofz = ofz.sort_values(["date", "tenor_years"])

    # Build a dict: date -> (tenors_array, yields_array) for fast lookup
    ofz_by_date: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]] = {}
    for dt, grp in ofz.groupby("date"):
        ofz_by_date[dt] = (grp["tenor_years"].values, grp["yield_pct"].values)

    trades = trades.copy()
    # Vectorized interpolation per date
    interp_results = np.full(len(trades), np.nan)
    for dt, idx in trades.groupby("date").groups.items():
        curve = ofz_by_date.get(dt)
        if curve is None:
            continue
        tenors, yields = curve
        durations = trades.loc[idx, "duration"].values
        interp_results[idx] = np.interp(durations, tenors, yields)

    trades["ofz_yield_interp"] = interp_results
    trades["g_spread"] = trades["yield_close"] - trades["ofz_yield_interp"]

    n_valid = trades["g_spread"].notna().sum()
    n_total = len(trades)
    logger.info(
        "G-spread computed: %d / %d valid (%.1f%%)",
        n_valid,
        n_total,
        100 * n_valid / n_total if n_total > 0 else 0,
    )

    trades.drop(columns=["ofz_yield_interp"], inplace=True)
    return trades


def compute_ofz_slope(ofz: pd.DataFrame) -> pd.DataFrame:
    """Compute OFZ slope = yield(10Y) - yield(2Y) per date.

    Returns DataFrame with columns: date, ofz_slope.
    """
    logger.info("Computing OFZ slope (10Y - 2Y)...")
    pivot = ofz.pivot_table(index="date", columns="tenor_years", values="yield_pct")

    if 10.0 not in pivot.columns or 2.0 not in pivot.columns:
        logger.warning(
            "OFZ curve missing 10Y or 2Y tenor. Available: %s",
            list(pivot.columns),
        )
        return pd.DataFrame({"date": pivot.index, "ofz_slope": np.nan})

    slope = pd.DataFrame({
        "date": pivot.index,
        "ofz_slope": pivot[10.0].values - pivot[2.0].values,
    })
    logger.info("OFZ slope computed: %d dates", len(slope))
    return slope


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add age_days, time_to_maturity, volume_ma20, is_state_owned, net_debt_ebitda."""
    logger.info("Adding derived features...")

    df["issue_date"] = pd.to_datetime(df["issue_date"], errors="coerce")
    df["maturity_date"] = pd.to_datetime(df["maturity_date"], errors="coerce")

    # age_days = date - issue_date
    df["age_days"] = (df["date"] - df["issue_date"]).dt.days

    # time_to_maturity = maturity_date - date (in days)
    df["time_to_maturity"] = (df["maturity_date"] - df["date"]).dt.days

    # volume_ma20: rolling 20-day mean of volume_rub per bond
    df = df.sort_values(["isin", "date"])
    df["volume_ma20"] = (
        df.groupby("isin")["volume_rub"]
        .transform(lambda s: s.rolling(window=20, min_periods=1).mean())
    )

    # is_state_owned: 1 if issuer_name matches any known state company
    if "issuer_name" in df.columns:
        pattern = "|".join(re.escape(c) for c in STATE_COMPANIES)
        df["is_state_owned"] = (
            df["issuer_name"]
            .fillna("")
            .str.contains(pattern, case=False, regex=True)
            .astype(int)
        )
    else:
        df["is_state_owned"] = 0

    logger.info("Derived features added.")
    return df


def merge_multipliers(panel: pd.DataFrame) -> pd.DataFrame:
    """Merge financial multipliers from Cbonds via merge_asof on available_date.

    Uses 90-day lag: report_date + 90 days = available_date.
    Joins by company_id (must be in panel from issuer_info).
    """
    if not MULTIPLIERS_FILE.exists():
        logger.warning("Multipliers file not found, setting net_debt_ebitda=NaN")
        panel["net_debt_ebitda"] = np.nan
        return panel

    logger.info("Loading multipliers from %s", MULTIPLIERS_FILE)
    mult = pd.read_csv(MULTIPLIERS_FILE, parse_dates=["available_date", "report_date"])
    logger.info("Multipliers loaded: %d rows, %d companies", len(mult), mult["company_id"].nunique())

    if "company_id" not in panel.columns:
        logger.warning("company_id missing from panel — cannot merge multipliers")
        panel["net_debt_ebitda"] = np.nan
        return panel

    # Select metrics to merge (keep only most useful)
    metric_cols = ["net_debt_ebitda", "ebitda_margin", "ebitda_yoy",
                   "revenue_yoy", "debt_to_equity", "roe", "roa"]
    available_cols = [c for c in metric_cols if c in mult.columns]
    mult_subset = mult[["company_id", "available_date"] + available_cols].copy()
    mult_subset = mult_subset.dropna(subset=["company_id", "available_date"])
    mult_subset["company_id"] = mult_subset["company_id"].astype(int)

    mult_subset = mult_subset.sort_values(["company_id", "available_date"])

    panel_sorted = panel.sort_values(["company_id", "date"]).copy()
    panel_sorted["company_id_int"] = pd.to_numeric(
        panel_sorted["company_id"], errors="coerce"
    )
    valid = panel_sorted["company_id_int"].notna()

    logger.info("Merging multipliers via merge_asof (90-day PIT lag)...")

    # merge_asof: for each (date, company_id), find latest report with available_date <= date
    valid_panel = panel_sorted[valid].copy()
    valid_panel["company_id_int"] = valid_panel["company_id_int"].astype(int)
    valid_panel = valid_panel.sort_values("date")

    merged = pd.merge_asof(
        valid_panel,
        mult_subset.sort_values("available_date"),
        left_on="date",
        right_on="available_date",
        left_by="company_id_int",
        right_by="company_id",
        direction="backward",
    )

    invalid_panel = panel_sorted[~valid].copy()
    for c in available_cols:
        invalid_panel[c] = np.nan

    result = pd.concat([merged, invalid_panel], ignore_index=True)
    result = result.drop(columns=["company_id_int", "available_date"], errors="ignore")
    if "company_id_y" in result.columns:
        result = result.drop(columns=["company_id_y"])
    if "company_id_x" in result.columns:
        result = result.rename(columns={"company_id_x": "company_id"})

    coverage = result["net_debt_ebitda"].notna().mean() * 100
    logger.info(
        "Multipliers merged: net_debt_ebitda coverage %.1f%% of panel rows", coverage
    )
    return result


def merge_rsbu(panel: pd.DataFrame) -> pd.DataFrame:
    """Merge РСБУ fundamentals from bo.nalog.gov.ru via merge_asof on available_date.

    Joins by issuer_inn. Keeps raw values + derived (net_debt, ebit, icr).
    """
    if not RSBU_FILE.exists():
        logger.warning("RSBU file not found, skipping fundamentals merge")
        return panel

    logger.info("Loading RSBU fundamentals from %s", RSBU_FILE)
    rsbu = pd.read_csv(RSBU_FILE, parse_dates=["available_date", "period_end_date"])
    logger.info("RSBU loaded: %d rows, %d companies", len(rsbu), rsbu["inn"].nunique())

    # Need issuer_inn from issuer_info — load it if not in panel
    if "issuer_inn" not in panel.columns:
        issuer = pd.read_csv(ISSUER_INFO_FILE, usecols=["isin", "issuer_inn"])
        issuer = issuer.drop_duplicates(subset="isin")
        panel = panel.merge(issuer, on="isin", how="left")
        logger.info("Added issuer_inn from issuer_info")

    rsbu_cols = ["cash", "long_debt", "short_debt", "profit_before_tax",
                 "interest_paid", "net_debt", "ebit", "icr", "total_assets", "revenue"]
    available = [c for c in rsbu_cols if c in rsbu.columns]
    rsbu_subset = rsbu[["inn", "available_date"] + available].copy()
    rsbu_subset = rsbu_subset.dropna(subset=["inn", "available_date"])
    rsbu_subset["inn"] = rsbu_subset["inn"].astype(str).str.strip()
    # Rename rsbu columns to avoid collisions
    rename_map = {c: f"rsbu_{c}" for c in available}
    rsbu_subset = rsbu_subset.rename(columns=rename_map)
    # Ensure one row per (inn, available_date) for merge_asof
    rsbu_subset = rsbu_subset.drop_duplicates(subset=["inn", "available_date"], keep="last")

    panel = panel.copy()
    panel["inn_str"] = panel["issuer_inn"].astype(str).str.replace(".0", "", regex=False).str.strip()
    valid = panel["inn_str"].notna() & (panel["inn_str"] != "nan") & (panel["inn_str"] != "")

    logger.info("Merging RSBU via merge_asof (by INN)...")
    valid_panel = panel[valid].sort_values("date").copy()
    rsbu_subset = rsbu_subset.sort_values("available_date")

    merged = pd.merge_asof(
        valid_panel,
        rsbu_subset,
        left_on="date",
        right_on="available_date",
        left_by="inn_str",
        right_by="inn",
        direction="backward",
    )

    invalid_panel = panel[~valid].copy()
    for c in rename_map.values():
        invalid_panel[c] = np.nan

    # Drop duplicated columns if any (from merge_asof artifacts)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    invalid_panel = invalid_panel.loc[:, ~invalid_panel.columns.duplicated()]

    all_cols = list(set(merged.columns) | set(invalid_panel.columns))
    for c in all_cols:
        if c not in merged.columns:
            merged[c] = np.nan
        if c not in invalid_panel.columns:
            invalid_panel[c] = np.nan

    result = pd.concat([merged[all_cols], invalid_panel[all_cols]], ignore_index=True)
    result = result.drop(columns=["inn_str", "available_date", "inn"], errors="ignore")

    coverage = result["rsbu_net_debt"].notna().mean() * 100
    logger.info("RSBU merged: net_debt coverage %.1f%% of panel rows", coverage)
    return result


def apply_liquidity_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter bonds by monthly liquidity criteria.

    Per bond-month:
      - Count days with volume_rub > 0 (active days)
      - Compute average volume_rub on active days
    Keep bond-months with >= MIN_ACTIVE_DAYS_PER_MONTH active days
    AND >= MIN_AVG_VOLUME_RUB average volume.
    """
    logger.info("Applying liquidity filter...")
    n_before = len(df)

    df = df.copy()
    df["ym"] = df["date"].dt.to_period("M")

    active = df[df["volume_rub"] > 0].groupby(["isin", "ym"]).agg(
        active_days=("volume_rub", "count"),
        avg_volume=("volume_rub", "mean"),
    )

    # Keep bond-months that pass both thresholds
    liquid = active[
        (active["active_days"] >= MIN_ACTIVE_DAYS_PER_MONTH)
        & (active["avg_volume"] >= MIN_AVG_VOLUME_RUB)
    ].reset_index()[["isin", "ym"]]

    # Inner merge to keep only liquid rows
    df = df.merge(liquid, on=["isin", "ym"], how="inner")
    df.drop(columns=["ym"], inplace=True)

    n_after = len(df)
    logger.info(
        "Liquidity filter: %d -> %d rows (removed %d, %.1f%%)",
        n_before,
        n_after,
        n_before - n_after,
        100 * (n_before - n_after) / n_before if n_before > 0 else 0,
    )
    return df


def build_panel(test_mode: bool, apply_liq_filter: bool) -> pd.DataFrame:
    """Run the full panel assembly pipeline."""

    # 1. Load all data sources
    trades = load_trades(test_mode)
    ofz = load_ofz_curve()
    macro = load_macro()
    universe = load_universe()
    issuer_info = load_issuer_info()
    ratings = load_ratings()

    # 2. Compute G-spread (target)
    trades = compute_g_spread(trades, ofz)

    # 3. Compute OFZ slope
    ofz_slope = compute_ofz_slope(ofz)

    # 4. Merge trades + universe (on isin)
    logger.info("Merging trades + universe...")
    panel = trades.merge(
        universe,
        on="isin",
        how="left",
    )
    logger.info("After universe merge: %d rows", len(panel))

    # 5. Merge + issuer_info (on isin)
    logger.info("Merging + issuer_info...")
    panel = panel.merge(
        issuer_info,
        on="isin",
        how="left",
    )
    logger.info("After issuer_info merge: %d rows", len(panel))

    # 6. Merge + macro (on date)
    logger.info("Merging + macro...")
    macro_cols = ["date", "key_rate", "usdrub", "imoex_return", "imoex_realized_vol", "rvi", "oil_brent"]
    macro_subset = macro[[c for c in macro_cols if c in macro.columns]].copy()
    panel = panel.merge(
        macro_subset,
        on="date",
        how="left",
    )
    logger.info("After macro merge: %d rows", len(panel))

    # 7. Merge + OFZ slope (on date)
    logger.info("Merging + OFZ slope...")
    panel = panel.merge(ofz_slope, on="date", how="left")
    logger.info("After OFZ slope merge: %d rows", len(panel))

    # 8. Merge + ratings via merge_asof (by isin, on date, backward)
    logger.info("Merging + ratings (merge_asof)...")
    panel = panel.sort_values("date")
    ratings = ratings.sort_values("date")
    panel = pd.merge_asof(
        panel,
        ratings,
        on="date",
        by="isin",
        direction="backward",
    )
    logger.info("After ratings merge: %d rows", len(panel))

    # 9. Add derived features
    panel = add_derived_features(panel)
    panel = merge_multipliers(panel)
    panel = merge_rsbu(panel)

    # 10. Apply liquidity filter
    if apply_liq_filter:
        panel = apply_liquidity_filter(panel)
    else:
        logger.info("Skipping liquidity filter (--no-liquidity-filter).")

    # 11. Drop rows where g_spread is NaN
    n_before = len(panel)
    panel = panel.dropna(subset=["g_spread"])
    logger.info(
        "Dropped %d rows with NaN g_spread (%d remaining)",
        n_before - len(panel),
        len(panel),
    )

    # 12. Select and order output columns
    # Use issuer_name from universe (not issuer_info company_name)
    available = [c for c in OUTPUT_COLUMNS if c in panel.columns]
    missing = [c for c in OUTPUT_COLUMNS if c not in panel.columns]
    if missing:
        logger.warning("Missing output columns (will be NaN): %s", missing)
        for col in missing:
            panel[col] = np.nan

    panel = panel[OUTPUT_COLUMNS].copy()

    panel = panel.sort_values(["date", "isin"]).reset_index(drop=True)

    return panel


def print_summary(df: pd.DataFrame) -> None:
    """Print a human-readable summary of the panel."""
    if df.empty:
        logger.info("Panel is empty.")
        return

    logger.info("--- Panel summary ---")
    logger.info("Total rows:          %d", len(df))
    logger.info("Unique dates:        %d", df["date"].nunique())
    logger.info("Unique ISINs:        %d", df["isin"].nunique())
    logger.info(
        "Date range:          %s .. %s",
        df["date"].min().strftime("%Y-%m-%d") if hasattr(df["date"].min(), "strftime") else df["date"].min(),
        df["date"].max().strftime("%Y-%m-%d") if hasattr(df["date"].max(), "strftime") else df["date"].max(),
    )
    logger.info("G-spread range:      %.2f .. %.2f", df["g_spread"].min(), df["g_spread"].max())
    logger.info("G-spread median:     %.2f", df["g_spread"].median())

    for col in ["duration", "rating_numeric", "key_rate", "sector", "ofz_slope"]:
        if col in df.columns:
            null_pct = df[col].isna().mean() * 100
            logger.info("Null rate %-16s: %.1f%%", col, null_pct)

    logger.info("Output file:         %s", OUTPUT_FILE)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the final modeling panel dataset."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=f"Test mode: use only the first {TEST_MODE_MONTHS} months of data.",
    )
    parser.add_argument(
        "--no-liquidity-filter",
        action="store_true",
        help="Skip the liquidity filter (keep all bond-months).",
    )
    args = parser.parse_args()

    panel = build_panel(
        test_mode=args.test,
        apply_liq_filter=not args.no_liquidity_filter,
    )

    panel.to_csv(OUTPUT_FILE, index=False)
    print_summary(panel)
    logger.info("Done.")


if __name__ == "__main__":
    main()
