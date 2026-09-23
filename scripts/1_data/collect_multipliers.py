"""
Collect quarterly financial multipliers from Cbonds.ru for all companies in issuer_info.csv.

Uses Playwright (sync API) with saved session cookies. Multipliers are fetched
via internal Cbonds JSON API endpoint ``/api/company/{id}/multipliers/``
(called through ``page.evaluate`` inside an authenticated browser session).

For each company, the API returns a table of financial metrics over time
(quarterly snapshots). The script parses the response, extracts selected
multipliers, and produces a long-format DataFrame that is then pivoted to
wide format.

Output file:
  - data/multipliers.csv -- one row per (company_id, report_date)

Columns: company_id, report_date, available_date,
         net_debt_ebitda, ebitda_margin, ebitda_yoy, revenue_yoy,
         debt_to_equity, roe, roa, p_e, p_s, p_b, capex_ratio

Usage:
    python scripts/collect_multipliers.py          # full collection
    python scripts/collect_multipliers.py --test   # first 5 companies only
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
from playwright.sync_api import (
    Browser,
    BrowserContext,
    Page,
    sync_playwright,
)
from tqdm import tqdm


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ISSUER_INFO_FILE = DATA_DIR / "issuer_info.csv"
COOKIES_FILE = DATA_DIR / "cbonds_cookies.json"
OUTPUT_FILE = DATA_DIR / "multipliers.csv"
CHECKPOINT_FILE = DATA_DIR / "multipliers_checkpoint.csv"

BASE_URL = "https://cbonds.ru"

UNSUPPORTED_COOKIE_FIELDS = {"partitionKey", "_crHasCrossSiteAncestor"}

DELAY_SECONDS = 1.0
CHECKPOINT_INTERVAL = 50
TEST_MODE_LIMIT = 5
PAGE_TIMEOUT = 30_000  # milliseconds

# Point-in-time lag: financial reports typically become available ~90 days
# after the reporting period ends.
PIT_LAG_DAYS = 90

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Metric name mapping: Russian label (or key phrase) -> English column name

# The "field" value in scroll_body rows is HTML like:
#   <span id="report-id-123" >Чистый долг / EBITDA
# We strip tags, then match the remaining text against these patterns.

METRIC_MAP: dict[str, str] = {
    "Чистый долг / EBITDA":            "net_debt_ebitda",
    "Чистый долг/EBITDA":              "net_debt_ebitda",
    "Net Debt / EBITDA":               "net_debt_ebitda",
    "Net Debt/EBITDA":                 "net_debt_ebitda",

    "Рентабельность по EBITDA (%)":    "ebitda_margin",
    "Рентабельность по EBITDA":        "ebitda_margin",
    "EBITDA Margin":                   "ebitda_margin",
    "EBITDA margin":                   "ebitda_margin",

    "EBITDA, YoY (%)":                 "ebitda_yoy",
    "EBITDA, YoY %":                   "ebitda_yoy",
    "EBITDA, YoY":                     "ebitda_yoy",
    "EBITDA YoY":                      "ebitda_yoy",

    "Выручка, YoY (%)":                "revenue_yoy",
    "Выручка, YoY %":                  "revenue_yoy",
    "Выручка, YoY":                    "revenue_yoy",
    "Revenue, YoY %":                  "revenue_yoy",
    "Revenue, YoY":                    "revenue_yoy",
    "Revenue YoY":                     "revenue_yoy",

    "Общий долг / Капитал":            "debt_to_equity",
    "Общий долг/Капитал":              "debt_to_equity",
    "Total Debt / Equity":             "debt_to_equity",
    "Total Debt/Equity":               "debt_to_equity",
    "Debt / Equity":                   "debt_to_equity",
    "Debt/Equity":                     "debt_to_equity",

    "ROE (%)":                         "roe",
    "ROA (%)":                         "roa",
    "ROE":                             "roe",
    "ROA":                             "roa",
    "P/E":                             "p_e",
    "P/S":                             "p_s",
    "P/B":                             "p_b",
    "P / E":                           "p_e",
    "P / S":                           "p_s",
    "P / B":                           "p_b",
    "P/FCF":                           "p_fcf",
    "EPS Basic":                       "eps_basic",

    "Коэффициент капитальных расходов": "capex_ratio",
    "Capex Ratio":                     "capex_ratio",
    "CAPEX ratio":                     "capex_ratio",
    "CAPEX Ratio":                     "capex_ratio",
    "Capex ratio":                     "capex_ratio",
    "CapEx Ratio":                     "capex_ratio",
}

# All target columns -- order matters for final CSV
TARGET_COLUMNS = [
    "net_debt_ebitda", "ebitda_margin", "ebitda_yoy", "revenue_yoy",
    "debt_to_equity", "roe", "roa", "p_e", "p_s", "p_b", "p_fcf",
    "eps_basic", "capex_ratio",
]


# Cookie handling (same pattern as collect_cbonds.py)


def load_cookies(path: Path) -> list[dict[str, object]]:
    """Load cookies from JSON, stripping Playwright-unsupported fields."""
    raw: list[dict[str, object]] = json.loads(path.read_text(encoding="utf-8"))
    cleaned = [
        {k: v for k, v in cookie.items() if k not in UNSUPPORTED_COOKIE_FIELDS}
        for cookie in raw
    ]
    logger.info("Loaded %d cookies from %s.", len(cleaned), path)
    return cleaned


# HTML / value parsing helpers

_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(html: str) -> str:
    """Remove HTML tags and return clean text."""
    return _TAG_RE.sub("", html).strip()


def parse_metric_name(field_html: str) -> str | None:
    """Extract metric name from the 'field' HTML string.

    The field looks like:
        '<span id="report-id-123" >Чистый долг / EBITDA'
    or sometimes:
        '<span id="report-id-123" >Чистый долг / EBITDA</span>'

    We strip tags, then look up in METRIC_MAP.
    """
    text = strip_html(field_html)
    if text in METRIC_MAP:
        return METRIC_MAP[text]
    # Try case-insensitive and stripped match
    text_lower = text.lower().strip()
    for key, val in METRIC_MAP.items():
        if key.lower().strip() == text_lower:
            return val
    return None


def parse_value(raw: str | int | float | None) -> float | None:
    """Parse a numeric value from the API response.

    Handles: empty strings, None, comma-as-decimal, percentage signs, etc.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if s in ("", "-", "—", "н/д", "n/a", "N/A"):
        return None
    # Remove percentage sign if present
    s = s.replace("%", "").strip()
    # Russian decimal comma -> dot
    s = s.replace(",", ".")
    # Remove spaces (thousand separator)
    s = s.replace(" ", "").replace("\u00a0", "")
    try:
        return float(s)
    except ValueError:
        return None


# API fetching (same pattern as collect_cbonds.py _fetch_api_json)

_JS_FETCH_TEMPLATE = """
async () => {{
    const resp = await fetch('{url}');
    if (!resp.ok) return {{error: resp.status}};
    return resp.json();
}}
"""


def fetch_multipliers_raw(page: Page, company_id: str) -> dict | None:
    """Fetch raw multipliers JSON for a company via in-browser fetch.

    Returns the parsed JSON dict on success, or None on failure.
    """
    url = f"/api/company/{company_id}/multipliers/"
    js = _JS_FETCH_TEMPLATE.format(url=url)
    try:
        data = page.evaluate(js)
    except Exception as exc:
        logger.debug("API call %s failed: %s", url, exc)
        return None

    if isinstance(data, dict):
        if "error" in data:
            logger.debug("API call %s returned HTTP %s", url, data["error"])
            return None
        return data

    logger.debug("API call %s returned unexpected type: %s", url, type(data))
    return None


def parse_multipliers(company_id: str, raw: dict) -> list[dict]:
    """Parse the multipliers API response into a list of flat records.

    Each record: {company_id, report_date, metric_eng, value}
    These will later be pivoted to wide format.
    """
    scroll_body = raw.get("scroll_body")
    scroll_head = raw.get("scroll_head")

    if not scroll_body or not scroll_head:
        logger.debug("Company %s: empty scroll_body or scroll_head.", company_id)
        return []

    # Build timestamp -> date_str mapping from scroll_head.
    # scroll_head is a list with one dict, e.g. [{"1609372800": "31.12.2020", ...}]
    ts_to_date: dict[str, str] = {}
    if isinstance(scroll_head, list) and len(scroll_head) > 0:
        head_dict = scroll_head[0] if isinstance(scroll_head[0], dict) else {}
        ts_to_date = {k: v for k, v in head_dict.items()}
    elif isinstance(scroll_head, dict):
        ts_to_date = {k: v for k, v in scroll_head.items()}

    if not ts_to_date:
        logger.debug("Company %s: could not parse scroll_head.", company_id)
        return []

    records: list[dict] = []

    for row in scroll_body:
        if not isinstance(row, dict):
            continue

        field_html = row.get("field", "")
        metric_eng = parse_metric_name(field_html)
        if metric_eng is None:
            # Not a metric we care about -- skip
            continue

        for ts_key, date_str in ts_to_date.items():
            if ts_key not in row:
                continue
            value = parse_value(row[ts_key])
            if value is None:
                continue
            records.append({
                "company_id": company_id,
                "report_date": date_str,
                "metric": metric_eng,
                "value": value,
            })

    return records


def records_to_wide(records: list[dict]) -> pd.DataFrame:
    """Convert long-format records to wide-format DataFrame.

    Input: list of {company_id, report_date, metric, value}
    Output: DataFrame with columns:
        company_id, report_date, available_date,
        net_debt_ebitda, ebitda_margin, ...
    """
    if not records:
        return pd.DataFrame(
            columns=["company_id", "report_date", "available_date"] + TARGET_COLUMNS
        )

    df_long = pd.DataFrame(records)

    # Pivot: rows = (company_id, report_date), columns = metric
    df_wide = (
        df_long
        .drop_duplicates(subset=["company_id", "report_date", "metric"], keep="last")
        .pivot_table(
            index=["company_id", "report_date"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )

    for col in TARGET_COLUMNS:
        if col not in df_wide.columns:
            df_wide[col] = None

    df_wide = df_wide[["company_id", "report_date"] + TARGET_COLUMNS]

    # Parse report_date and compute available_date (PIT lag)
    df_wide["report_date_parsed"] = pd.to_datetime(
        df_wide["report_date"], format="%d.%m.%Y", errors="coerce"
    )
    df_wide["available_date"] = (
        df_wide["report_date_parsed"] + timedelta(days=PIT_LAG_DAYS)
    ).dt.strftime("%Y-%m-%d")
    # Standardize report_date to YYYY-MM-DD
    df_wide["report_date"] = df_wide["report_date_parsed"].dt.strftime("%Y-%m-%d")
    df_wide.drop(columns=["report_date_parsed"], inplace=True)

    df_wide = df_wide[
        ["company_id", "report_date", "available_date"] + TARGET_COLUMNS
    ]

    return df_wide


def get_unique_company_ids(path: Path) -> list[str]:
    """Read issuer_info.csv and return sorted list of unique numeric company_ids."""
    df = pd.read_csv(path, dtype=str)
    ids = df["company_id"].dropna().unique()

    valid: list[str] = []
    for cid in ids:
        cid = str(cid).strip()
        if not cid or cid == "0":
            continue
        try:
            int(cid)
            valid.append(cid)
        except ValueError:
            logger.debug("Skipping non-numeric company_id: %r", cid)
            continue

    valid = sorted(set(valid), key=int)
    return valid


def save_checkpoint(all_records: list[dict], path: Path) -> None:
    """Save current accumulated records as wide-format CSV."""
    df = records_to_wide(all_records)
    df.to_csv(path, index=False)
    logger.info("Checkpoint saved: %d rows to %s", len(df), path)


def collect(
    company_ids: list[str],
    cookies: list[dict[str, object]],
) -> list[dict]:
    """Run the full collection pipeline.

    Opens a VISIBLE browser. On first launch, navigates to cbonds.ru
    so you can verify you're logged in before scraping starts.

    Returns list of long-format records:
        [{company_id, report_date, metric, value}, ...]
    """
    all_records: list[dict] = []
    succeeded = 0
    failed = 0
    empty = 0

    with sync_playwright() as pw:
        browser: Browser = pw.chromium.launch(headless=False)
        context: BrowserContext = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            locale="ru-RU",
        )
        if cookies:
            try:
                context.add_cookies(cookies)
            except Exception:
                logger.warning("Could not load cookies, continuing without them.")

        page: Page = context.new_page()
        page.set_default_timeout(PAGE_TIMEOUT)

        page.goto(BASE_URL, timeout=60_000)
        try:
            page.wait_for_load_state("networkidle", timeout=15_000)
        except Exception:
            pass
        input(
            "Browser opened. Verify you're logged in to Cbonds, "
            "then press Enter here..."
        )

        for idx, company_id in enumerate(
            tqdm(company_ids, desc="Companies", unit="co")
        ):
            try:
                raw = fetch_multipliers_raw(page, company_id)
                if raw is None:
                    logger.warning(
                        "Company %s: API returned no data (HTTP error or empty).",
                        company_id,
                    )
                    failed += 1
                    time.sleep(DELAY_SECONDS)
                    continue

                records = parse_multipliers(company_id, raw)
                if not records:
                    logger.debug(
                        "Company %s: no relevant multiplier rows found.",
                        company_id,
                    )
                    empty += 1
                else:
                    all_records.extend(records)
                    succeeded += 1
                    logger.debug(
                        "Company %s: extracted %d data points.",
                        company_id,
                        len(records),
                    )

            except Exception as exc:
                logger.warning(
                    "Company %s: unexpected error: %s", company_id, exc
                )
                failed += 1

            time.sleep(DELAY_SECONDS)

            if (idx + 1) % CHECKPOINT_INTERVAL == 0:
                logger.info(
                    "Checkpoint at company %d/%d. "
                    "Succeeded: %d, Empty: %d, Failed: %d",
                    idx + 1,
                    len(company_ids),
                    succeeded,
                    empty,
                    failed,
                )
                save_checkpoint(all_records, CHECKPOINT_FILE)

        browser.close()

    logger.info("=" * 60)
    logger.info("COLLECTION SUMMARY")
    logger.info("=" * 60)
    logger.info("Total companies:       %d", len(company_ids))
    logger.info("Succeeded (with data): %d", succeeded)
    logger.info("Empty (no metrics):    %d", empty)
    logger.info("Failed (errors):       %d", failed)
    logger.info("Total data points:     %d", len(all_records))
    logger.info("=" * 60)

    return all_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect financial multipliers from Cbonds.ru."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=f"Test mode: process only the first {TEST_MODE_LIMIT} companies.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load unique company_ids from issuer_info.csv
    company_ids = get_unique_company_ids(ISSUER_INFO_FILE)
    logger.info("Found %d unique company_ids in %s.", len(company_ids), ISSUER_INFO_FILE)

    if not company_ids:
        logger.error("No valid company_ids found. Exiting.")
        return

    if args.test:
        company_ids = company_ids[:TEST_MODE_LIMIT]
        logger.info("TEST MODE: processing only %d companies.", len(company_ids))

    cookies: list[dict[str, object]] = []
    if COOKIES_FILE.exists():
        cookies = load_cookies(COOKIES_FILE)
    else:
        logger.warning(
            "Cookies file not found at %s. Will rely on manual login.",
            COOKIES_FILE,
        )

    all_records = collect(company_ids, cookies)

    if not all_records:
        logger.warning("No data collected. Output file will not be created.")
        return

    df = records_to_wide(all_records)
    df.to_csv(OUTPUT_FILE, index=False)
    logger.info("Saved %d rows to %s", len(df), OUTPUT_FILE)

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Removed checkpoint: %s", CHECKPOINT_FILE)

    logger.info("Sample output (first 5 rows):")
    logger.info("\n%s", df.head().to_string())


if __name__ == "__main__":
    main()
