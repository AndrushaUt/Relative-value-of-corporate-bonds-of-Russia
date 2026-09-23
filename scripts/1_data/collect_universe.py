"""
Collect the universe of Russian corporate bonds from ISS MOEX API (TQCB board).

Fetches all bonds, applies filters (RUB-only, fixed coupon, min issue volume,
no structural products), enriches with emitter_id and issue_date from auxiliary
MOEX endpoints, and saves the filtered universe to data/universe.csv.

WARNING -- SURVIVORSHIP BIAS:
    The TQCB board only lists bonds that are **currently traded**.  Bonds that
    have already matured or been delisted are NOT included.  This means the
    resulting universe suffers from survivorship bias, which can distort
    downstream analyses (e.g. default-rate estimation, yield modeling).

    TODO: Augment with historical bonds via the ``/iss/history/`` endpoint
    or by periodically snapshotting the universe over time.

Usage:
    python scripts/collect_universe.py          # full run
    python scripts/collect_universe.py --test   # first page only, no enrichment
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

TQCB_URL = (
    "https://iss.moex.com/iss/engines/stock/markets/bonds/boards/TQCB/securities.json"
)

# Columns available on the TQCB board endpoint.  ISSUEDATE and EMITTER_ID are
# NOT available here -- they are fetched from auxiliary endpoints after filtering.
TQCB_COLUMNS = (
    "SECID,ISIN,SHORTNAME,SECNAME,FACEVALUE,FACEUNIT,"
    "COUPONPERIOD,COUPONVALUE,COUPONPERCENT,MATDATE,"
    "ISSUESIZE,SECTYPE,LISTLEVEL"
)

TQCB_PARAMS = {
    "iss.meta": "off",
    "iss.only": "securities",
    "securities.columns": TQCB_COLUMNS,
}

# Bulk securities index endpoint -- provides emitent_id for all securities
SECURITIES_INDEX_URL = "https://iss.moex.com/iss/securities.json"

# Per-security description endpoint (provides ISSUEDATE, EMITTER_ID, INN, etc.)
SECURITY_DESC_URL = "https://iss.moex.com/iss/securities/{secid}.json"

PAGE_SIZE = 100  # MOEX ISS default page size
REQUEST_DELAY = 0.1  # seconds between requests
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0  # seconds

MIN_ISSUE_VOLUME_RUB = 100_000_000

# Column rename map: internal name -> output snake_case name
RENAME_MAP = {
    "ISIN": "isin",
    "SECID": "ticker",
    "SHORTNAME": "short_name",
    "SECNAME": "issuer_name",
    "emitter_id": "emitter_id",
    "issue_date": "issue_date",
    "issuer_inn": "issuer_inn",
    "MATDATE": "maturity_date",
    "COUPONPERCENT": "coupon_rate",
    "COUPONPERIOD": "coupon_period",
    "COUPONVALUE": "coupon_value",
    "FACEVALUE": "face_value",
    "ISSUESIZE": "issue_size",
    "LISTLEVEL": "list_level",
    "SECTYPE": "sec_type",
}

OUTPUT_COLUMNS = [
    "isin",
    "ticker",
    "short_name",
    "issuer_name",
    "emitter_id",
    "issuer_inn",
    "issue_date",
    "maturity_date",
    "coupon_rate",
    "coupon_period",
    "coupon_frequency",
    "coupon_value",
    "face_value",
    "issue_size",
    "issue_volume_rub",
    "list_level",
    "sec_type",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def fetch_json(
    session: requests.Session,
    url: str,
    params: Optional[Dict] = None,
    max_retries: int = MAX_RETRIES,
    initial_backoff: float = INITIAL_BACKOFF,
) -> dict:
    """Fetch JSON from *url* with exponential-backoff retries.

    Retries on network errors and HTTP 429 (Too Many Requests).
    Raises immediately on other HTTP errors (4xx/5xx).
    """
    backoff = initial_backoff
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=30)

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", backoff))
                logger.warning(
                    "HTTP 429 -- Retry-After=%.1fs (attempt %d/%d)",
                    retry_after,
                    attempt,
                    max_retries,
                )
                time.sleep(retry_after)
                backoff *= 2
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.HTTPError:
            logger.error(
                "HTTP %d for %s (attempt %d/%d)",
                resp.status_code,
                url,
                attempt,
                max_retries,
            )
            raise

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ) as exc:
            logger.warning(
                "Network error: %s (attempt %d/%d, backoff %.1fs)",
                exc,
                attempt,
                max_retries,
                backoff,
            )
            if attempt == max_retries:
                raise
            time.sleep(backoff)
            backoff *= 2

    raise RuntimeError(f"Failed to fetch {url} after {max_retries} retries")


def fetch_tqcb_bonds(
    session: requests.Session,
    test_mode: bool = False,
) -> pd.DataFrame:
    """Fetch all bonds from TQCB board with pagination.

    The MOEX ISS API may return all rows in a single page for some column
    selections, but we paginate defensively (start=0, 100, 200, ...) and
    stop when an empty page is returned or fewer rows than PAGE_SIZE.

    In test mode only the first page is fetched.
    """
    all_rows: List[list] = []
    columns: Optional[List[str]] = None
    start = 0

    logger.info("Fetching bonds from TQCB board...")
    pbar = tqdm(desc="Fetching TQCB pages", unit="page")

    # MOEX TQCB board returns ALL bonds in a single response
    # (the `start` parameter is ignored for this endpoint).
    # No pagination needed — one request fetches everything.
    data = fetch_json(session, TQCB_URL, TQCB_PARAMS)

    securities = data.get("securities", {})
    columns = securities.get("columns", [])
    all_rows = securities.get("data", [])

    pbar.update(1)
    pbar.set_postfix(bonds=len(all_rows))
    pbar.close()

    if columns is None or not all_rows:
        raise RuntimeError(
            "MOEX TQCB endpoint returned no data. "
            "Check network connectivity and endpoint availability."
        )

    df = pd.DataFrame(all_rows, columns=columns)
    logger.info("Fetched %d bonds from TQCB board", len(df))
    return df


def fetch_emitter_ids(
    session: requests.Session,
    secids: List[str],
) -> Dict[str, int]:
    """Fetch emitent_id for a set of SECIDs via the bulk securities index.

    Uses ``/iss/securities.json`` with pagination.
    Returns ``{SECID: emitent_id}``.
    """
    logger.info("Fetching emitter IDs from securities index...")
    result: Dict[str, int] = {}
    start = 0
    pbar = tqdm(desc="Fetching emitter IDs", unit="page")
    target = set(secids)

    while True:
        params = {
            "iss.meta": "off",
            "iss.only": "securities",
            "securities.columns": "secid,emitent_id",
            "engine": "stock",
            "market": "bonds",
            "start": start,
        }
        data = fetch_json(session, SECURITIES_INDEX_URL, params)
        rows = data.get("securities", {}).get("data", [])

        if not rows:
            break

        for row in rows:
            sid, eid = row[0], row[1]
            if sid in target:
                result[sid] = eid

        pbar.update(1)
        pbar.set_postfix(found=len(result))

        if len(rows) < PAGE_SIZE:
            break

        start += PAGE_SIZE
        time.sleep(REQUEST_DELAY)

    pbar.close()
    logger.info("Fetched emitter_id for %d / %d bonds", len(result), len(secids))
    return result


def fetch_issue_details(
    session: requests.Session,
    secids: List[str],
) -> Dict[str, Dict[str, str]]:
    """Fetch ISSUEDATE and INN for each SECID via per-security description endpoint.

    Returns ``{SECID: {"issue_date": "YYYY-MM-DD", "issuer_inn": "..."}}``.
    """
    logger.info("Fetching issue details (date + INN) for %d bonds...", len(secids))
    result: Dict[str, Dict[str, str]] = {}

    for secid in tqdm(secids, desc="Fetching issue details", unit="bond"):
        url = SECURITY_DESC_URL.format(secid=secid)
        params = {"iss.meta": "off", "iss.only": "description"}
        try:
            data = fetch_json(session, url, params)
            desc_rows = data.get("description", {}).get("data", [])
            details: Dict[str, str] = {}
            for row in desc_rows:
                if row[0] == "ISSUEDATE":
                    details["issue_date"] = row[2]
                elif row[0] == "INN":
                    details["issuer_inn"] = row[2]
            if details:
                result[secid] = details
        except Exception as exc:
            logger.warning("Failed to fetch description for %s: %s", secid, exc)

        time.sleep(REQUEST_DELAY)

    n_dates = sum(1 for v in result.values() if "issue_date" in v)
    n_inns = sum(1 for v in result.values() if "issuer_inn" in v)
    logger.info(
        "Fetched issue_date for %d / %d, issuer_inn for %d / %d bonds",
        n_dates, len(secids), n_inns, len(secids),
    )
    return result


def filter_universe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all universe filters and log statistics at each step."""
    total_start = len(df)
    logger.info("=" * 60)
    logger.info("FILTERING: Starting with %d bonds", total_start)
    logger.info("=" * 60)

    # 1. RUB only (FACEUNIT == 'SUR')
    before = len(df)
    df = df[df["FACEUNIT"] == "SUR"].copy()
    removed = before - len(df)
    logger.info(
        "  FACEUNIT == 'SUR' (RUB only):        %d -> %d  (-%d)",
        before, len(df), removed,
    )

    # 2. Fixed coupon (COUPONPERCENT > 0 and not null)
    before = len(df)
    df = df[df["COUPONPERCENT"].notna() & (df["COUPONPERCENT"] > 0)].copy()
    removed = before - len(df)
    logger.info(
        "  COUPONPERCENT > 0 (fixed coupon):     %d -> %d  (-%d)",
        before, len(df), removed,
    )

    # 2b. Exclude floaters -- COUPONPERCENT > 0 does NOT reliably filter them
    #     because floaters also carry a positive current coupon rate.
    #     Heuristic: exclude bonds whose SHORTNAME contains typical floater markers.
    _FLOATER_MARKERS = ("\u0424\u041b\u0422", "\u0444\u043b\u043e\u0430\u0442", "float", "\u041a\u0421+", "RUONIA")
    before = len(df)
    floater_pattern = "|".join(_FLOATER_MARKERS)
    floater_mask = df["SHORTNAME"].str.contains(
        floater_pattern, case=False, na=False,
    )
    n_floaters = int(floater_mask.sum())
    df = df[~floater_mask].copy()
    logger.info(
        "  Exclude floaters (name heuristic):    %d -> %d  (-%d)",
        before, len(df), n_floaters,
    )

    # 3. Issue volume >= 100 M RUB
    before = len(df)
    df["issue_volume_rub"] = df["ISSUESIZE"] * df["FACEVALUE"]
    df = df[df["issue_volume_rub"] >= MIN_ISSUE_VOLUME_RUB].copy()
    removed = before - len(df)
    logger.info(
        "  issue_volume >= 100M RUB:             %d -> %d  (-%d)",
        before, len(df), removed,
    )

    # 4. Exclude structural products
    before = len(df)
    structural_mask = (
        df["SHORTNAME"].str.contains(r"(?i)\u0441\u0442\u0440\u0443\u043a\u0442\u0443\u0440\u043d", na=False)
        | df["SECNAME"].str.contains(r"(?i)\u0441\u0442\u0440\u0443\u043a\u0442\u0443\u0440\u043d", na=False)
    )
    n_structural = int(structural_mask.sum())
    df = df[~structural_mask].copy()
    logger.info(
        "  Exclude structural products:          %d -> %d  (-%d)",
        before, len(df), n_structural,
    )

    logger.info("=" * 60)
    logger.info(
        "FILTERING COMPLETE: %d -> %d bonds (removed %d total)",
        total_start, len(df), total_start - len(df),
    )
    logger.info("=" * 60)
    return df


def enrich_and_prepare(
    session: requests.Session,
    df: pd.DataFrame,
    skip_enrichment: bool = False,
) -> pd.DataFrame:
    """Add emitter_id, issue_date, issuer_inn; rename columns; select output columns."""

    secids = df["SECID"].tolist()

    if skip_enrichment:
        logger.info("Skipping enrichment (test mode)")
        df = df.copy()
        df["emitter_id"] = None
        df["issue_date"] = None
        df["issuer_inn"] = None
    else:
        df = df.copy()

        # Fetch emitter IDs (bulk, paginated -- fast)
        eid_map = fetch_emitter_ids(session, secids)
        df["emitter_id"] = df["SECID"].map(eid_map)
        n_missing = int(df["emitter_id"].isna().sum())
        if n_missing:
            logger.warning("emitter_id missing for %d bonds", n_missing)

        # Fetch issue dates + INN (per-security description -- slower)
        details_map = fetch_issue_details(session, secids)
        df["issue_date"] = df["SECID"].map(
            lambda s: details_map.get(s, {}).get("issue_date")
        )
        df["issuer_inn"] = df["SECID"].map(
            lambda s: details_map.get(s, {}).get("issuer_inn")
        )
        n_missing = int(df["issue_date"].isna().sum())
        if n_missing:
            logger.warning("issue_date missing for %d bonds", n_missing)
        n_missing_inn = int(df["issuer_inn"].isna().sum())
        if n_missing_inn:
            logger.warning("issuer_inn missing for %d bonds", n_missing_inn)

    # Ensure issue_volume_rub exists (should from filter step)
    if "issue_volume_rub" not in df.columns:
        df["issue_volume_rub"] = df["ISSUESIZE"] * df["FACEVALUE"]

    # Derive coupon_frequency (times per year) from coupon_period (days)
    df["coupon_frequency"] = (365 / df["COUPONPERIOD"]).round().astype("Int64")

    df = df.rename(columns=RENAME_MAP)

    df = df[OUTPUT_COLUMNS].copy()

    # Sort by issue volume descending for convenience
    df = df.sort_values("issue_volume_rub", ascending=False).reset_index(drop=True)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect universe of Russian corporate bonds from MOEX ISS API"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: fetch only first page, skip enrichment",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "universe.csv"

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "kursach-bond-collector/1.0 (academic research)"}
    )

    # Survivorship bias warning
    logger.warning(
        "SURVIVORSHIP BIAS: TQCB board only contains currently traded bonds. "
        "Matured/delisted bonds are missing. See module docstring for details."
    )

    try:
        # 1. Fetch all bonds from TQCB board
        df_raw = fetch_tqcb_bonds(session, test_mode=args.test)

        # 2. Apply filters
        df_filtered = filter_universe(df_raw)

        # 3. Enrich with emitter_id and issue_date, prepare output
        df_out = enrich_and_prepare(
            session, df_filtered, skip_enrichment=args.test
        )

        # 4. Save
        df_out.to_csv(output_path, index=False)
        logger.info("Saved %d bonds to %s", len(df_out), output_path)

        print()
        print("=" * 60)
        print(f"  Total bonds fetched:    {len(df_raw):>6}")
        print(f"  After all filters:      {len(df_out):>6}")
        print(f"  Saved to:               {output_path}")
        if args.test:
            print("  Mode:                   TEST (first page only, no enrichment)")
        print("=" * 60)
        print()

        print("Top 10 bonds by issue volume:")
        preview_cols = ["ticker", "short_name", "coupon_rate", "issue_volume_rub"]
        print(df_out[preview_cols].head(10).to_string(index=False))

    except requests.exceptions.RequestException as exc:
        logger.error("MOEX API request failed: %s", exc)
        logger.error("Aborting. Check your network connection and try again.")
        raise SystemExit(1)
    except RuntimeError as exc:
        logger.error("%s", exc)
        raise SystemExit(1)
    finally:
        session.close()


if __name__ == "__main__":
    main()
