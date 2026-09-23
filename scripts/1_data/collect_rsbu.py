"""
Collect RSBU financial statements from bo.nalog.gov.ru for bond issuers
missing Cbonds multipliers.

Fetches balance-sheet and income-statement line items from the Russian Federal
Tax Service public API (RSBU filings).  Targets ~156 corporate issuers whose
data is not available through Cbonds.

Output: data/fundamentals_rsbu.csv

Usage:
    python scripts/collect_rsbu.py           # full collection
    python scripts/collect_rsbu.py --test    # first 5 INNs only
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
from datetime import timedelta
from pathlib import Path

import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm as atqdm


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUTPUT_FILE = DATA_DIR / "fundamentals_rsbu.csv"
ISSUER_FILE = DATA_DIR / "issuer_info.csv"
MULTIPLIERS_FILE = DATA_DIR / "multipliers.csv"

BASE_URL = "https://bo.nalog.gov.ru"
SEARCH_URL = f"{BASE_URL}/advanced-search/organizations"
BFO_LIST_URL = f"{BASE_URL}/nbo/organizations/{{org_id}}/bfo"
BFO_DETAILS_URL = f"{BASE_URL}/nbo/bfo/{{report_id}}/details"

MAX_CONCURRENT = 10
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
TEST_MODE_COUNT = 5

# Municipal / government entity filter.
# Matches region names (область, республика, край, округ) as whole words,
# plus known city-level municipal issuers and non-corporate entries.
REGION_PATTERN = re.compile(
    r"\b(область|Республика|край|округ)\b",
    re.IGNORECASE,
)
MUNICIPAL_CITIES = frozenset([
    "Москва",
    "Санкт-Петербург",
    "Новосибирск",
    "Томск",
    "Красноярск",
    "Казань",
    "Екатеринбург",
    "Нижний Новгород",
])
OTHER_EXCLUDE_PATTERNS = [
    re.compile(r"^Россия:", re.IGNORECASE),
]

# RSBU row codes to extract.
# Balance sheet (form 1): current year value uses prefix "current",
# prior year uses "previous".
BALANCE_FIELDS = {
    "1250": "cash",              # Денежные средства и эквиваленты
    "1410": "long_debt",         # Долгосрочные заёмные средства
    "1510": "short_debt",        # Краткосрочные заёмные средства
    "1600": "total_assets",      # Баланс (актив)
}

# Income statement (form 2).
FINANCIAL_RESULT_FIELDS = {
    "2110": "revenue",            # Выручка
    "2300": "profit_before_tax",  # Прибыль (убыток) до налогообложения
    "2330": "interest_paid",      # Проценты к уплате
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def is_municipal(name: str) -> bool:
    """Return True if the company name looks like a municipal/government entity."""
    if pd.isna(name):
        return False
    name_stripped = name.strip()
    if name_stripped in MUNICIPAL_CITIES:
        return True
    if REGION_PATTERN.search(name_stripped):
        return True
    for pat in OTHER_EXCLUDE_PATTERNS:
        if pat.search(name_stripped):
            return True
    return False


def load_target_inns() -> pd.DataFrame:
    """Load INNs for all companies with valid issuer_inn.

    Returns a DataFrame with columns: inn (str), company_id (int), company_name (str).
    One row per unique company_id. Excludes only municipalities/regions.
    """
    info = pd.read_csv(ISSUER_FILE)

    # Filter: has INN, not a duplicate company_id
    mask = info["issuer_inn"].notna() & info["company_id"].notna()
    candidates = info.loc[mask].drop_duplicates("company_id").copy()

    # Convert INN from float to string (10-digit, no decimals)
    candidates["inn"] = candidates["issuer_inn"].astype(int).astype(str)
    # Some INNs might be 12 digits (individual entrepreneurs) -- keep all
    candidates = candidates[candidates["inn"].str.len().between(10, 12)]

    # Filter out municipal / government entities
    before = len(candidates)
    candidates = candidates[~candidates["company_name"].apply(is_municipal)]
    filtered_out = before - len(candidates)

    logger.info(
        "Target INNs: %d total, %d municipal filtered out, %d remaining",
        before, filtered_out, len(candidates),
    )

    return candidates[["inn", "company_id", "company_name"]].reset_index(drop=True)


async def fetch_json(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url: str,
    *,
    params: dict | None = None,
    max_retries: int = MAX_RETRIES,
) -> dict | list | None:
    """Fetch JSON with exponential backoff and semaphore throttling.

    Returns parsed JSON (dict or list) on success, or None on failure.
    """
    for attempt in range(max_retries):
        try:
            async with semaphore:
                async with session.get(url, params=params) as resp:
                    if resp.status == 429:
                        wait = 2 ** attempt
                        logger.warning(
                            "HTTP 429 rate-limited. Waiting %ds (attempt %d/%d)",
                            wait, attempt + 1, max_retries,
                        )
                        await asyncio.sleep(wait)
                        continue

                    if resp.status == 404:
                        return None

                    resp.raise_for_status()
                    return await resp.json(content_type=None)

        except (
            aiohttp.ClientResponseError,
            aiohttp.ClientError,
            asyncio.TimeoutError,
        ) as exc:
            wait = 2 ** attempt
            logger.warning(
                "Request %s failed: %s. Retrying in %ds (%d/%d)",
                url.split("/")[-1][:40], exc, wait, attempt + 1, max_retries,
            )
            await asyncio.sleep(wait)

    logger.error("All %d retries exhausted for %s", max_retries, url)
    return None


async def search_org(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    inn: str,
) -> int | None:
    """Search for an organization by INN. Returns org_id or None."""
    data = await fetch_json(
        session, semaphore,
        SEARCH_URL,
        params={"inn": inn, "page": "0"},
    )
    if data is None:
        return None

    content = data.get("content", [])
    if not content:
        return None

    return content[0].get("id")


async def list_reports(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    org_id: int,
) -> list[dict]:
    """Get the list of BFO reports for an organization.

    Returns list of report dicts with keys: id, period, actualBfoDate, etc.
    """
    data = await fetch_json(
        session, semaphore,
        BFO_LIST_URL.format(org_id=org_id),
    )
    if data is None or not isinstance(data, list):
        return []
    return data


def extract_fields(details: list[dict], report_year: str) -> dict | None:
    """Extract target financial fields from a report details response.

    The API returns a list; we take the first element.  Balance and financial-
    result dicts use keys like "current1250", "previous1250", etc.  We extract
    the "current" prefix values for the reporting year.

    Returns a flat dict of extracted fields, or None if data is missing.
    """
    if not details or not isinstance(details, list):
        return None

    detail = details[0]
    balance = detail.get("balance") or {}
    fin_result = detail.get("financialResult") or {}

    row: dict[str, object] = {"report_year": int(report_year)}

    # Extract balance-sheet fields (current-period values)
    for code, col_name in BALANCE_FIELDS.items():
        key = f"current{code}"
        val = balance.get(key)
        # Values are in thousands of rubles or raw rubles (depends on company
        # reporting unit).  Store as-is; we normalize downstream if needed.
        row[col_name] = val

    # Extract income-statement fields (current-period values)
    for code, col_name in FINANCIAL_RESULT_FIELDS.items():
        key = f"current{code}"
        val = fin_result.get(key)
        row[col_name] = val

    # Check that we got at least some non-null data
    data_cols = list(BALANCE_FIELDS.values()) + list(FINANCIAL_RESULT_FIELDS.values())
    if all(row.get(c) is None for c in data_cols):
        return None

    return row


async def fetch_report_details(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    report_id: int,
    report_year: str,
    actual_bfo_date: str | None,
) -> dict | None:
    """Fetch and parse a single report's details.

    Returns a dict with extracted fields plus metadata, or None on failure.
    """
    data = await fetch_json(
        session, semaphore,
        BFO_DETAILS_URL.format(report_id=report_id),
    )
    if data is None:
        return None

    row = extract_fields(data, report_year)
    if row is None:
        return None

    # period_end_date: annual RSBU = Dec 31 of the reporting year
    row["period_end_date"] = f"{report_year}-12-31"

    # available_date: use actualBfoDate if present, else period_end + 90 days
    if actual_bfo_date:
        row["available_date"] = actual_bfo_date[:10]  # strip time if present
    else:
        end_date = pd.Timestamp(f"{report_year}-12-31")
        row["available_date"] = (end_date + timedelta(days=90)).strftime("%Y-%m-%d")

    return row


async def process_issuer(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    inn: str,
    company_id: float,
    company_name: str,
) -> list[dict]:
    """Full pipeline for one issuer: search -> list reports -> fetch details.

    Returns a list of record dicts (one per report year).
    """
    # Step 1: search by INN
    org_id = await search_org(session, semaphore, inn)
    if org_id is None:
        return []

    # Step 2: list reports
    reports = await list_reports(session, semaphore, org_id)
    if not reports:
        return []

    # Step 3: fetch details for each report
    records: list[dict] = []
    for report in reports:
        report_id = report["id"]
        report_year = str(report.get("period", ""))
        actual_bfo_date = report.get("actualBfoDate")

        row = await fetch_report_details(
            session, semaphore, report_id, report_year, actual_bfo_date,
        )
        if row is None:
            continue

        row["inn"] = inn
        row["company_id"] = int(company_id)
        row["company_name"] = company_name
        row["org_id"] = org_id

        records.append(row)

    return records


async def collect_all(targets: pd.DataFrame) -> list[dict]:
    """Run the async collection for all target issuers."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)

    stats = {"searched": 0, "found": 0, "reports": 0, "errors": 0}

    async with aiohttp.ClientSession(
        headers={
            "User-Agent": "kursach-bond-rsbu-collector/1.0 (academic research)",
            "Accept": "application/json",
        },
        timeout=timeout,
    ) as session:

        async def _process_one(row: pd.Series) -> list[dict]:
            inn = row["inn"]
            company_id = row["company_id"]
            company_name = row["company_name"]

            stats["searched"] += 1
            try:
                records = await process_issuer(
                    session, semaphore, inn, company_id, company_name,
                )
            except Exception as exc:
                logger.error("Unhandled error for INN %s (%s): %s", inn, company_name, exc)
                stats["errors"] += 1
                return []

            if records:
                stats["found"] += 1
                stats["reports"] += len(records)

            return records

        tasks = [
            asyncio.create_task(
                _process_one(row),
                name=f"inn-{row['inn']}",
            )
            for _, row in targets.iterrows()
        ]

        all_records: list[dict] = []
        for coro in atqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Issuers",
            unit="co",
        ):
            records = await coro
            all_records.extend(records)

    logger.info(
        "--- Collection stats ---\n"
        "  Searched:  %d\n"
        "  Found:     %d\n"
        "  Reports:   %d\n"
        "  Errors:    %d\n"
        "  Not found: %d",
        stats["searched"],
        stats["found"],
        stats["reports"],
        stats["errors"],
        stats["searched"] - stats["found"] - stats["errors"],
    )

    return all_records


def compute_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Compute net_debt, EBIT, and ICR from raw RSBU fields."""
    # Coerce to numeric (some values might come as strings or None)
    numeric_cols = [
        "cash", "long_debt", "short_debt", "total_assets",
        "revenue", "profit_before_tax", "interest_paid",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # net_debt = long_debt + short_debt - cash
    df["net_debt"] = df["long_debt"].fillna(0) + df["short_debt"].fillna(0) - df["cash"].fillna(0)

    # EBIT = profit_before_tax + interest_paid
    df["ebit"] = df["profit_before_tax"].fillna(0) + df["interest_paid"].fillna(0)

    # ICR = EBIT / interest_paid (only when interest_paid > 0)
    df["icr"] = None
    mask = df["interest_paid"].notna() & (df["interest_paid"] > 0)
    df.loc[mask, "icr"] = df.loc[mask, "ebit"] / df.loc[mask, "interest_paid"]

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect RSBU financial statements from bo.nalog.gov.ru",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=f"Test mode: process only the first {TEST_MODE_COUNT} INNs.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    targets = load_target_inns()
    if targets.empty:
        logger.error("No target INNs found. Aborting.")
        return

    if args.test:
        targets = targets.head(TEST_MODE_COUNT)
        logger.info(
            "TEST MODE: processing only %d INNs: %s",
            len(targets), targets["inn"].tolist(),
        )

    logger.info(
        "Starting RSBU collection for %d corporate issuers", len(targets),
    )

    all_records = asyncio.run(collect_all(targets))

    if not all_records:
        logger.warning("No data collected. Check API availability / INN coverage.")
        return

    df = pd.DataFrame(all_records)

    df["company_id"] = df["company_id"].astype(int)
    df["report_year"] = df["report_year"].astype(int)

    df = compute_derived_fields(df)

    output_cols = [
        "inn", "company_id", "company_name", "org_id",
        "report_year", "period_end_date", "available_date",
        "cash", "long_debt", "short_debt", "total_assets",
        "revenue", "profit_before_tax", "interest_paid",
        "net_debt", "ebit", "icr",
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    df = df[output_cols]

    df.sort_values(["company_id", "report_year"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(OUTPUT_FILE, index=False)

    logger.info("--- Output summary ---")
    logger.info("Total rows:        %d", len(df))
    logger.info("Unique companies:  %d", df["company_id"].nunique())
    logger.info("Unique INNs:       %d", df["inn"].nunique())
    logger.info("Year range:        %s .. %s", df["report_year"].min(), df["report_year"].max())
    logger.info("Null rates:")
    for col in ["cash", "long_debt", "short_debt", "profit_before_tax", "interest_paid"]:
        if col in df.columns:
            null_pct = df[col].isna().mean() * 100
            logger.info("  %-20s %.1f%%", col, null_pct)

    if "icr" in df.columns:
        icr_valid = df["icr"].notna()
        if icr_valid.any():
            logger.info(
                "ICR: %d values, median=%.2f, min=%.2f, max=%.2f",
                icr_valid.sum(),
                df.loc[icr_valid, "icr"].median(),
                df.loc[icr_valid, "icr"].min(),
                df.loc[icr_valid, "icr"].max(),
            )

    logger.info("Output file:       %s", OUTPUT_FILE)
    logger.info("Done.")


if __name__ == "__main__":
    main()
