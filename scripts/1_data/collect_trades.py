"""
Collect daily trade data for corporate bonds from MOEX ISS API.

Fetches historical trade data from the TQCB and EQOB boards (corporate bonds)
for all securities in the project universe.  Days are fetched in parallel using
asyncio + aiohttp; pages within a single day are fetched sequentially (cursor
dependency).  A semaphore caps concurrency to stay well under MOEX rate limits.

Output: data/trades_daily.csv with columns:
    date, isin, close_price, yield_close, volume_rub, num_trades, duration

Usage:
    python scripts/collect_trades.py          # full collection
    python scripts/collect_trades.py --test   # first 5 trading days only
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date, timedelta
from pathlib import Path

import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm as atqdm


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUTPUT_FILE = DATA_DIR / "trades_daily.csv"
CHECKPOINT_FILE = DATA_DIR / "trades_daily_checkpoint.csv"
UNIVERSE_FILE = DATA_DIR / "universe.csv"

# MOEX switched boards: EQOB (before ~Dec 2019) -> TQCB (from ~Dec 2019).
# Both boards overlap in Dec 2019 - mid 2020. We query both and deduplicate.
BOARDS = ["TQCB", "EQOB"]
BASE_URL_TEMPLATE = (
    "https://iss.moex.com/iss/history/engines/stock/markets/bonds"
    "/boards/{board}/securities.json"
)

DATE_START = "2019-01-01"
DATE_END = "2026-03-30"

MAX_CONCURRENT = 15        # semaphore limit (MOEX allows ~50/sec)
REQUEST_TIMEOUT = 30       # seconds (per-request via session timeout)
MAX_RETRIES = 5            # exponential backoff retries
TEST_MODE_DAYS = 5         # number of days to fetch in --test mode
CHECKPOINT_INTERVAL = 200  # save partial results every N days
PAGE_SIZE = 100            # MOEX ISS default page size

CLOSE_PRICE_MIN = 10.0
CLOSE_PRICE_MAX = 200.0
YIELD_CLOSE_MIN = -5.0
YIELD_CLOSE_MAX = 50.0

# Column mapping from MOEX ISS names to our schema
# NOTE: MOEX "VOLUME" = lot count, "VALUE" = ruble volume.
COLUMN_MAP = {
    "TRADEDATE": "date",
    "SECID": "isin",
    "CLOSE": "close_price",
    "YIELDCLOSE": "yield_close",
    "VALUE": "volume_rub",
    "NUMTRADES": "num_trades",
    "DURATION": "duration",
}

HISTORY_COLUMNS = ",".join(COLUMN_MAP.keys())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


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
        if current.weekday() < 5:
            days.append(current.isoformat())
        current += timedelta(days=1)
    return days


def load_universe_isins() -> set[str]:
    """Load the set of ISINs from universe.csv for O(1) membership checks."""
    df = pd.read_csv(UNIVERSE_FILE, usecols=["isin"])
    isins = set(df["isin"].dropna().unique())
    logger.info("Loaded %d ISINs from universe.", len(isins))
    return isins


def parse_cursor(data: dict) -> tuple[int, int]:
    """Parse history.cursor block to extract (index, total).

    The cursor block has columns [INDEX, TOTAL, PAGESIZE] with one data row.
    Returns (current_index, total_rows).
    """
    cursor_block = data.get("history.cursor", {})
    columns = cursor_block.get("columns", [])
    rows = cursor_block.get("data", [])

    if not rows or not columns:
        return 0, 0

    row = dict(zip(columns, rows[0]))
    return int(row.get("INDEX", 0)), int(row.get("TOTAL", 0))


def parse_history_rows(data: dict) -> tuple[list[str], list[list]]:
    """Extract column names and data rows from the history block.

    Returns (columns, rows).
    """
    history_block = data.get("history", {})
    columns = history_block.get("columns", [])
    rows = history_block.get("data", [])
    return columns, rows


def save_checkpoint(records: list[dict], path: Path) -> None:
    """Save accumulated records to a checkpoint CSV."""
    if not records:
        return
    df = pd.DataFrame(records)
    df.rename(columns=COLUMN_MAP, inplace=True)
    df.to_csv(path, index=False)
    logger.info("Checkpoint saved: %d rows -> %s", len(records), path)


def run_sanity_checks(df: pd.DataFrame) -> None:
    """Warn about values outside plausible ranges and check for duplicates."""
    if df.empty:
        return

    if "close_price" in df.columns:
        bad_close = df["close_price"].notna() & (
            (df["close_price"] < CLOSE_PRICE_MIN)
            | (df["close_price"] > CLOSE_PRICE_MAX)
        )
        if (n_bad_close := bad_close.sum()) > 0:
            examples = df.loc[bad_close, ["date", "isin", "close_price"]].head(5)
            logger.warning(
                "%d rows with close_price outside [%.0f, %.0f]. Examples:\n%s",
                n_bad_close, CLOSE_PRICE_MIN, CLOSE_PRICE_MAX,
                examples.to_string(index=False),
            )

    if "yield_close" in df.columns:
        bad_yield = df["yield_close"].notna() & (
            (df["yield_close"] < YIELD_CLOSE_MIN)
            | (df["yield_close"] > YIELD_CLOSE_MAX)
        )
        if (n_bad_yield := bad_yield.sum()) > 0:
            examples = df.loc[bad_yield, ["date", "isin", "yield_close"]].head(5)
            logger.warning(
                "%d rows with yield_close outside [%.0f, %.0f]. Examples:\n%s",
                n_bad_yield, YIELD_CLOSE_MIN, YIELD_CLOSE_MAX,
                examples.to_string(index=False),
            )

    dupes = df.duplicated(subset=["date", "isin"], keep=False)
    if (n_dupes := dupes.sum()) > 0:
        examples = df.loc[dupes, ["date", "isin"]].head(10)
        logger.warning(
            "%d rows are duplicates on (date, isin). Examples:\n%s",
            n_dupes, examples.to_string(index=False),
        )
    else:
        logger.info("No duplicate (date, isin) pairs found.")


def print_summary(df: pd.DataFrame) -> None:
    """Print a human-readable summary of the collected data."""
    if df.empty:
        logger.info("No data collected.")
        return

    n_rows = len(df)
    n_dates = df["date"].nunique()
    n_isins = df["isin"].nunique()
    date_min = df["date"].min()
    date_max = df["date"].max()

    yield_null_rate = df["yield_close"].isna().mean() * 100
    duration_null_rate = df["duration"].isna().mean() * 100

    logger.info("--- Collection summary ---")
    logger.info("Total rows:       %d", n_rows)
    logger.info("Unique dates:     %d", n_dates)
    logger.info("Unique ISINs:     %d", n_isins)
    logger.info("Date range:       %s .. %s", date_min, date_max)
    logger.info("Null rate yield:  %.2f%%", yield_null_rate)
    logger.info("Null rate duration: %.2f%%", duration_null_rate)
    logger.info("Output file:      %s", OUTPUT_FILE)


async def fetch_json(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url: str,
    params: dict,
    max_retries: int = MAX_RETRIES,
) -> dict | None:
    """Fetch JSON from MOEX ISS with exponential backoff.

    Acquires a semaphore slot before each HTTP request.  Handles HTTP 429
    (rate limit) and transient errors by retrying up to ``max_retries``
    times with exponential backoff (1, 2, 4, 8, 16 s).

    Returns parsed JSON dict on success, or None if all retries exhausted.
    """
    for attempt in range(max_retries):
        try:
            async with semaphore:
                async with session.get(url, params=params) as resp:
                    if resp.status == 429:
                        wait = 2 ** attempt
                        logger.warning(
                            "HTTP 429 rate-limited. Waiting %d s (attempt %d/%d)",
                            wait, attempt + 1, max_retries,
                        )
                        await asyncio.sleep(wait)
                        continue

                    resp.raise_for_status()
                    return await resp.json(content_type=None)

        except (aiohttp.ClientResponseError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            wait = 2 ** attempt
            logger.warning(
                "Request failed: %s. Retrying in %d s (attempt %d/%d)",
                exc, wait, attempt + 1, max_retries,
            )
            await asyncio.sleep(wait)

    logger.error("All %d retries exhausted for %s", max_retries, url)
    return None


async def fetch_day(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    day_str: str,
    universe_isins: set[str],
    is_first: bool = False,
) -> list[dict]:
    """Fetch all trade rows for a single day, paginating as needed.

    Returns a list of dicts (one per filtered row) with MOEX column names.
    Only rows whose SECID is in ``universe_isins`` are kept.

    Pages within a board are fetched sequentially (cursor dependency).
    Both boards (TQCB, EQOB) are fetched sequentially to keep logic simple.
    """
    base_params: dict = {
        "date": day_str,
        "iss.meta": "off",
        "iss.only": "history,history.cursor",
        "history.columns": HISTORY_COLUMNS,
        "start": 0,
    }

    all_rows: list[list] = []
    columns: list[str] | None = None

    for board in BOARDS:
        board_url = BASE_URL_TEMPLATE.format(board=board)
        board_params = {**base_params, "start": 0}

        data = await fetch_json(session, semaphore, board_url, board_params)
        if data is None:
            continue

        board_cols, rows = parse_history_rows(data)
        if not board_cols or not rows:
            continue

        # On the very first successful fetch, log actual columns
        if is_first and columns is None:
            logger.info("MOEX history columns returned: %s", board_cols)

        columns = board_cols
        _, total = parse_cursor(data)
        all_rows.extend(rows)

        # Fetch remaining pages for this board (sequential -- cursor dependency)
        if total > PAGE_SIZE:
            for start in range(PAGE_SIZE, total, PAGE_SIZE):
                page_params = {**board_params, "start": start}
                page_data = await fetch_json(session, semaphore, board_url, page_params)
                if page_data is None:
                    logger.warning(
                        "Day %s board %s: page start=%d failed.",
                        day_str, board, start,
                    )
                    continue
                _, page_rows = parse_history_rows(page_data)
                all_rows.extend(page_rows)

    if columns is None or not all_rows:
        return []

    # Filter by universe ISINs and deduplicate (same bond on both boards)
    secid_idx = columns.index("SECID") if "SECID" in columns else None
    if secid_idx is None:
        logger.error("SECID column not found in response for day %s.", day_str)
        return []

    seen: set[str] = set()
    filtered: list[dict] = []
    for row in all_rows:
        secid = row[secid_idx]
        if secid in universe_isins and secid not in seen:
            seen.add(secid)
            filtered.append(dict(zip(columns, row)))

    return filtered


async def collect_all(
    trading_days: list[str],
    universe_isins: set[str],
    test_mode: bool,
) -> list[dict]:
    """Fetch trade data for all trading days concurrently.

    Days are dispatched in parallel (limited by semaphore).  Checkpoints
    are saved every CHECKPOINT_INTERVAL completed days.
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)

    async with aiohttp.ClientSession(
        headers={"User-Agent": "kursach-bond-collector/1.0 (academic research)"},
        timeout=timeout,
    ) as session:
        tasks: list[asyncio.Task[list[dict]]] = []
        for i, day_str in enumerate(trading_days):
            is_first = i == 0
            task = asyncio.create_task(
                fetch_day(session, semaphore, day_str, universe_isins, is_first=is_first),
                name=f"day-{day_str}",
            )
            tasks.append(task)

        all_records: list[dict] = []
        days_completed = 0

        for coro in atqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Days",
            unit="day",
        ):
            day_records = await coro
            all_records.extend(day_records)
            days_completed += 1

            # Checkpoint every N days
            if days_completed % CHECKPOINT_INTERVAL == 0 and all_records:
                save_checkpoint(all_records, CHECKPOINT_FILE)

    return all_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect daily corporate bond trades from MOEX ISS API."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=f"Test mode: fetch only the first {TEST_MODE_DAYS} trading days.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    universe_isins = load_universe_isins()
    if not universe_isins:
        logger.error("No ISINs loaded from universe.csv. Aborting.")
        return

    trading_days = generate_trading_days(DATE_START, DATE_END)
    if args.test:
        trading_days = trading_days[:TEST_MODE_DAYS]
        logger.info(
            "TEST MODE: processing only %d days (%s .. %s)",
            len(trading_days), trading_days[0], trading_days[-1],
        )

    logger.info(
        "Collecting trades: %s .. %s (%d weekdays to query)",
        DATE_START, DATE_END, len(trading_days),
    )

    all_records = asyncio.run(collect_all(trading_days, universe_isins, args.test))

    if not all_records:
        logger.error("No data collected. Check network / API availability.")
        return

    df = pd.DataFrame(all_records)
    df.rename(columns=COLUMN_MAP, inplace=True)

    df["date"] = df["date"].astype(str)
    for col in ("close_price", "yield_close", "volume_rub", "duration"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "num_trades" in df.columns:
        df["num_trades"] = pd.to_numeric(df["num_trades"], errors="coerce").astype("Int64")

    # Deduplicate (same bond from EQOB + TQCB overlap period)
    before_dedup = len(df)
    df.drop_duplicates(subset=["date", "isin"], keep="last", inplace=True)
    if len(df) < before_dedup:
        logger.info("Removed %d duplicate (date, isin) rows.", before_dedup - len(df))

    df.sort_values(["date", "isin"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    run_sanity_checks(df)

    df.to_csv(OUTPUT_FILE, index=False)
    print_summary(df)

    # Clean up checkpoint if final save succeeded
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Removed checkpoint file after successful save.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
