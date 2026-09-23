"""
Collect OFZ zero-coupon yield curve (KBD) from MOEX ISS API.

Fetches the government bond yield curve data from the MOEX ZCYC (zero-coupon
yield curve) endpoint for the period 2019-01-01 to 2026-03-30.  Each trading
day is fetched individually because the endpoint accepts only a single `date`
parameter, not a date range.

The resulting dataset contains daily zero-coupon yields at various tenors,
used as the risk-free curve for G-spread calculation.

Output: data/ofz_curve.csv with columns (date, tenor_years, yield_pct)

Usage:
    python scripts/collect_ofz_curve.py          # full collection
    python scripts/collect_ofz_curve.py --test    # first 5 trading days only
"""

import argparse
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUTPUT_FILE = DATA_DIR / "ofz_curve.csv"

BASE_URL = "https://iss.moex.com/iss/engines/stock/zcyc.json"

DATE_START = "2019-01-01"
DATE_END = "2026-03-30"

REQUEST_DELAY = 0.05      # seconds between requests
REQUEST_TIMEOUT = 30      # seconds
MAX_RETRIES = 5           # exponential backoff retries
TEST_MODE_DAYS = 5        # number of days to fetch in --test mode
YIELD_MIN = 0.0           # minimum plausible yield, %
YIELD_MAX = 30.0          # maximum plausible yield, %

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def make_session() -> requests.Session:
    """Create a requests.Session with connection pooling."""
    session = requests.Session()
    session.headers["User-Agent"] = (
        "kursach-bond-collector/1.0 (academic research)"
    )
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=4,
        pool_maxsize=4,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_json(
    session: requests.Session,
    url: str,
    params: dict,
    max_retries: int = MAX_RETRIES,
) -> dict | None:
    """Fetch JSON from MOEX ISS with exponential backoff.

    Handles HTTP 429 (rate limit) and transient errors by retrying up to
    ``max_retries`` times with exponential backoff (1, 2, 4, 8, 16 s).

    Returns parsed JSON dict on success, or None if all retries exhausted.
    """
    for attempt in range(max_retries):
        try:
            resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)

            if resp.status_code == 429:
                wait = 2 ** attempt
                logger.warning(
                    "HTTP 429 rate-limited. Waiting %d s (attempt %d/%d)",
                    wait, attempt + 1, max_retries,
                )
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.RequestException as exc:
            wait = 2 ** attempt
            logger.warning(
                "Request failed: %s. Retrying in %d s (attempt %d/%d)",
                exc, wait, attempt + 1, max_retries,
            )
            time.sleep(wait)

    logger.error("All %d retries exhausted for %s", max_retries, url)
    return None


def generate_trading_days(start: str, end: str) -> list[str]:
    """Generate a list of weekday date strings (YYYY-MM-DD) in [start, end].

    MOEX is closed on weekends; public holidays are handled by skipping
    days that return empty data from the API.
    """
    dt_start = date.fromisoformat(start)
    dt_end = date.fromisoformat(end)
    days: list[str] = []
    current = dt_start
    while current <= dt_end:
        # Monday=0 .. Friday=4 are weekdays
        if current.weekday() < 5:
            days.append(current.isoformat())
        current += timedelta(days=1)
    return days


def fetch_day(
    session: requests.Session,
    day_str: str,
) -> pd.DataFrame | None:
    """Fetch yield curve data for a single trading day.

    Returns a DataFrame with columns from the ``yearyields`` block,
    or None if the day has no data (holiday) or the request failed.
    """
    params = {
        "date": day_str,
        "iss.meta": "off",
        "iss.only": "yearyields",
    }

    data = fetch_json(session, BASE_URL, params)
    if data is None:
        logger.warning("Day %s failed after retries; skipping.", day_str)
        return None

    yy_block = data.get("yearyields", {})
    columns = yy_block.get("columns", [])
    rows = yy_block.get("data", [])

    if not rows or not columns:
        return None

    return pd.DataFrame(rows, columns=columns)


def validate_yields(df: pd.DataFrame) -> pd.DataFrame:
    """Warn about yield values outside the plausible range [0, 30]%.

    Does not remove outliers -- only logs warnings so the user can inspect.
    Returns the dataframe unchanged.
    """
    if df.empty:
        return df

    outliers = df[
        (df["yield_pct"] < YIELD_MIN) | (df["yield_pct"] > YIELD_MAX)
    ]
    if not outliers.empty:
        n = len(outliers)
        examples = outliers.head(5)[["date", "tenor_years", "yield_pct"]]
        logger.warning(
            "%d yield values outside [%.1f, %.1f]%% range. Examples:\n%s",
            n, YIELD_MIN, YIELD_MAX, examples.to_string(index=False),
        )
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print a human-readable summary of the collected data."""
    if df.empty:
        logger.info("No data collected.")
        return

    n_dates = df["date"].nunique()
    n_rows = len(df)
    date_min = df["date"].min()
    date_max = df["date"].max()
    tenors = sorted(df["tenor_years"].unique())

    logger.info("--- Collection summary ---")
    logger.info("Unique dates:   %d", n_dates)
    logger.info("Total rows:     %d", n_rows)
    logger.info("Date range:     %s .. %s", date_min, date_max)
    logger.info("Tenors (years): %s", tenors)
    logger.info("Output file:    %s", OUTPUT_FILE)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect OFZ zero-coupon yield curve from MOEX ISS API."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=f"Test mode: fetch only the first {TEST_MODE_DAYS} trading days.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    trading_days = generate_trading_days(DATE_START, DATE_END)
    if args.test:
        trading_days = trading_days[:TEST_MODE_DAYS]
        logger.info(
            "TEST MODE: processing only %d days (%s .. %s)",
            len(trading_days), trading_days[0], trading_days[-1],
        )

    logger.info(
        "Collecting OFZ curve: %s .. %s (%d weekdays to query)",
        DATE_START, DATE_END, len(trading_days),
    )

    session = make_session()
    try:
        frames: list[pd.DataFrame] = []

        for day_str in tqdm(trading_days, desc="Days", unit="day"):
            df_day = fetch_day(session, day_str)
            if df_day is not None and not df_day.empty:
                frames.append(df_day)
            time.sleep(REQUEST_DELAY)

        if not frames:
            logger.error("No data collected. Check network / API availability.")
            return

        df_raw = pd.concat(frames, ignore_index=True)

        # Rename to target schema
        # MOEX ZCYC yearyields columns (lowercase):
        #   tradedate, tradetime, period, value
        rename_map = {
            "tradedate": "date",
            "period": "tenor_years",
            "value": "yield_pct",
        }

        expected_cols = set(rename_map.keys())
        available = [c for c in rename_map if c in df_raw.columns]
        missing = expected_cols - set(available)
        if missing:
            logger.error(
                "Missing required columns: %s. Got: %s",
                sorted(missing), list(df_raw.columns),
            )
            # Save raw dump for debugging
            debug_path = DATA_DIR / "ofz_curve_raw_debug.csv"
            df_raw.to_csv(debug_path, index=False)
            logger.error("Raw data saved to %s for inspection.", debug_path)
            return

        df = df_raw[available].rename(columns=rename_map).copy()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        nat_count = df["date"].isna().sum()
        if nat_count:
            logger.warning("Dropped %d rows with null/malformed tradedate.", nat_count)
            df.dropna(subset=["date"], inplace=True)
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")

        df["tenor_years"] = pd.to_numeric(df["tenor_years"], errors="coerce")
        df["yield_pct"] = pd.to_numeric(df["yield_pct"], errors="coerce")

        # Drop rows where tenor or yield is missing (shouldn't happen, but safe)
        before = len(df)
        df.dropna(subset=["tenor_years", "yield_pct"], inplace=True)
        dropped = before - len(df)
        if dropped:
            logger.warning("Dropped %d rows with missing tenor/yield.", dropped)

        df.sort_values(["date", "tenor_years"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        validate_yields(df)

        df.to_csv(OUTPUT_FILE, index=False)
        print_summary(df)
        logger.info("Done.")
    finally:
        session.close()


if __name__ == "__main__":
    main()
