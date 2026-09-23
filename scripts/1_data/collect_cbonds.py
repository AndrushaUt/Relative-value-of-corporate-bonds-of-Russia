"""
Collect credit ratings and issuer info from Cbonds.ru for bonds in universe.csv.

Uses Playwright (sync API) with saved session cookies. Ratings are fetched
via internal Cbonds JSON API endpoints (called through ``page.evaluate``
inside an authenticated browser session), eliminating fragile HTML parsing.

Optimization: ISINs are grouped by emitter_id (MOEX) so that the expensive
autocomplete search is done only once per unique emitter (~409 groups)
instead of once per ISIN (~1793). The Cbonds company_id and issuer data
fetched for one ISIN are applied to every ISIN in the same emitter group.

Output files:
  - data/ratings.csv     -- one row per rating action per bond's issuer
  - data/issuer_info.csv -- one row per bond (isin -> issuer mapping)

Usage:
    python scripts/collect_cbonds.py          # full collection (all emitters)
    python scripts/collect_cbonds.py --test   # first 5 emitters only
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
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
UNIVERSE_FILE = DATA_DIR / "universe.csv"
COOKIES_FILE = DATA_DIR / "cbonds_cookies.json"
OUTPUT_RATINGS = DATA_DIR / "ratings.csv"
OUTPUT_ISSUER_INFO = DATA_DIR / "issuer_info.csv"
CHECKPOINT_RATINGS = DATA_DIR / "ratings_checkpoint.csv"
CHECKPOINT_ISSUER_INFO = DATA_DIR / "issuer_info_checkpoint.csv"

BASE_URL = "https://cbonds.ru"

# Playwright fields that are not supported by the add_cookies API
UNSUPPORTED_COOKIE_FIELDS = {"partitionKey", "_crHasCrossSiteAncestor"}

DELAY_MIN = 1.0  # seconds between page loads
DELAY_MAX = 2.0
CHECKPOINT_INTERVAL = 50  # save every N emitters processed
TEST_MODE_LIMIT = 5
PAGE_TIMEOUT = 30_000  # milliseconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BondResult:
    """Data scraped from a single bond page."""

    isin: str
    company_id: str | None = None
    company_name: str | None = None
    error: str | None = None


@dataclass(slots=True)
class IssuerResult:
    """Data fetched for a single issuer/company."""

    company_id: str
    company_name: str = ""
    inn: str = ""
    sector: str = ""
    current_ratings: list[dict[str, str]] = field(default_factory=list)
    rating_history: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None


def load_cookies(path: Path) -> list[dict[str, object]]:
    """Load cookies from JSON, stripping Playwright-unsupported fields."""
    raw: list[dict[str, object]] = json.loads(path.read_text(encoding="utf-8"))
    cleaned = [
        {k: v for k, v in cookie.items() if k not in UNSUPPORTED_COOKIE_FIELDS}
        for cookie in raw
    ]
    logger.info("Loaded %d cookies from %s.", len(cleaned), path)
    return cleaned


def polite_delay() -> None:
    """Sleep a random interval to avoid hammering the server."""
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))


# Scraping: bond search (autocomplete) — kept from original, works well


def search_bond(page: Page, isin: str) -> str | None:
    """Search for an ISIN on Cbonds using the autocomplete search bar.

    Strategy:
      1. Go to cbonds.ru (if not already there)
      2. Type ISIN into the top search input
      3. Wait for autocomplete dropdown with results
      4. Click the first bond result (link containing /bonds/)
      5. Return the resulting bond page URL
    """
    try:
        if "cbonds.ru" not in page.url:
            page.goto(BASE_URL, timeout=PAGE_TIMEOUT, wait_until="domcontentloaded")

        search_input = page.query_selector(
            'input[type="search"], input[name="search"], '
            'input[placeholder*="поиск" i], header input[type="text"], '
            ".header input, input.search-input"
        )
        if not search_input:
            for inp in page.query_selector_all(
                "header input, nav input, .top-bar input, input"
            ):
                inp_type = (inp.get_attribute("type") or "").lower()
                if inp_type not in ("hidden", "submit", "checkbox", "radio"):
                    search_input = inp
                    break

        if not search_input:
            logger.warning("ISIN %s: could not find search input on page.", isin)
            return None

        search_input.click()
        search_input.fill("")
        search_input.type(isin, delay=50)

        # Wait for autocomplete dropdown to appear
        time.sleep(2)

        # Look for bond link in the dropdown
        bond_link = page.query_selector('a[href*="/bonds/"]')
        if bond_link:
            href = bond_link.get_attribute("href")
            bond_link.click()
            try:
                page.wait_for_load_state("domcontentloaded", timeout=10000)
            except Exception:
                pass
            if "/bonds/" in page.url:
                return page.url
            if href:
                return BASE_URL + href if href.startswith("/") else href

        logger.warning("ISIN %s: no bond found in autocomplete dropdown.", isin)
        return None

    except Exception as exc:
        logger.warning("ISIN %s: search failed: %s", isin, exc)
        return None


# Scraping: bond page — extract company_id from issuer link


def scrape_bond_page(page: Page, isin: str, bond_url: str) -> BondResult:
    """Navigate to a bond page and extract the company_id from the issuer link."""
    result = BondResult(isin=isin)

    try:
        page.goto(bond_url, timeout=PAGE_TIMEOUT, wait_until="domcontentloaded")
    except Exception as exc:
        result.error = f"Failed to load bond page: {exc}"
        logger.warning("ISIN %s: %s", isin, result.error)
        return result

    # Extract issuer company link — look for /company/{id}/
    company_link = page.query_selector(
        'a[href*="/company/"], a[href*="/organisations/"]'
    )
    if company_link:
        href = company_link.get_attribute("href") or ""
        if m := re.search(r"/(company|organisations)/(\d+)/?", href):
            result.company_id = m.group(2)
        result.company_name = (company_link.inner_text() or "").strip()

    return result


# Issuer: fetch ratings via internal JSON API

_JS_FETCH_TEMPLATE = """
async () => {{
    const resp = await fetch('{url}');
    if (!resp.ok) return {{error: resp.status}};
    return resp.json();
}}
"""


def _fetch_api_json(page: Page, url: str) -> list[dict[str, str]] | None:
    """Call an internal Cbonds API endpoint via page.evaluate(fetch(...)).

    Returns parsed JSON (list of dicts) on success, or None on failure.
    The fetch runs inside the authenticated browser session, so cookies
    are sent automatically.
    """
    js = _JS_FETCH_TEMPLATE.format(url=url)
    try:
        data = page.evaluate(js)
    except Exception as exc:
        logger.debug("API call %s failed: %s", url, exc)
        return None

    if isinstance(data, dict) and "error" in data:
        logger.debug("API call %s returned HTTP %s", url, data["error"])
        return None

    if isinstance(data, list):
        return data

    logger.debug("API call %s returned unexpected type: %s", url, type(data))
    return None


def _fetch_current_ratings(page: Page, company_id: str) -> list[dict[str, str]]:
    """Fetch current ratings from /api/company/{id}/ratings/."""
    url = f"/api/company/{company_id}/ratings/"
    raw = _fetch_api_json(page, url)
    if not raw:
        return []

    results: list[dict[str, str]] = []
    for entry in raw:
        results.append({
            "agency": entry.get("agency_name", ""),
            "rating": entry.get("scale_point_name", ""),
            "date_assigned": entry.get("rating_date", ""),
            "outlook": entry.get("forecast_name", ""),
            "scale": entry.get("scale_name", ""),
        })
    return results


def _fetch_rating_history(page: Page, company_id: str) -> list[dict[str, str]]:
    """Fetch full rating history from /api/company/{id}/ratings_history/."""
    url = f"/api/company/{company_id}/ratings_history/"
    raw = _fetch_api_json(page, url)
    if not raw:
        return []

    results: list[dict[str, str]] = []
    for entry in raw:
        results.append({
            "agency": entry.get("agency_name", ""),
            "rating": entry.get("scale_point_name", ""),
            "date_assigned": entry.get("date", ""),
            "outlook": entry.get("forecast_name", ""),
            "scale": entry.get("scale_name", ""),
        })
    return results


# Issuer: parse INN / sector / name from company page text


def _extract_company_info(page: Page, result: IssuerResult) -> None:
    """Extract company_name, INN, and sector from the company page."""
    # Company name: first line of h1 text (h1 may contain sub-elements)
    h1 = page.query_selector("h1")
    if h1:
        h1_text = (h1.inner_text() or "").strip()
        result.company_name = h1_text.split("\n")[0].strip()

    body_text = page.inner_text("body") or ""

    # INN: regex after ИНН label
    if inn_match := re.search(r"ИНН\s*(\d{10,12})", body_text):
        result.inn = inn_match.group(1)

    # Sector: the line AFTER the line containing "ОТРАСЛЬ"
    lines = body_text.split("\n")
    for i, line in enumerate(lines):
        if "ОТРАСЛЬ" in line.upper():
            for subsequent in lines[i + 1 :]:
                stripped = subsequent.strip()
                if stripped:
                    result.sector = stripped
                    break
            break


# Issuer: orchestrate company data collection


def scrape_issuer(page: Page, company_id: str) -> IssuerResult:
    """Collect all issuer data: navigate to company page for INN/sector,
    then call API endpoints for current ratings and rating history.
    """
    result = IssuerResult(company_id=company_id)
    company_url = f"{BASE_URL}/company/{company_id}/"

    # Navigate to company page for INN, sector, company name
    try:
        page.goto(company_url, timeout=PAGE_TIMEOUT, wait_until="domcontentloaded")
    except Exception as exc:
        result.error = f"Failed to load company page: {exc}"
        logger.warning("Company %s: %s", company_id, result.error)
        return result

    _extract_company_info(page, result)

    # Fetch ratings via API (browser is authenticated, fetch uses session cookies)
    result.current_ratings = _fetch_current_ratings(page, company_id)
    result.rating_history = _fetch_rating_history(page, company_id)

    logger.info(
        "Company %s (%s): INN=%s, sector=%s, %d current ratings, %d history entries",
        company_id,
        result.company_name,
        result.inn or "?",
        result.sector or "?",
        len(result.current_ratings),
        len(result.rating_history),
    )

    return result


def save_results(
    issuer_rows: list[dict[str, str]],
    rating_rows: list[dict[str, str]],
    issuer_path: Path,
    ratings_path: Path,
) -> None:
    """Save current results to CSV files."""
    if issuer_rows:
        pd.DataFrame(issuer_rows).to_csv(issuer_path, index=False)
        logger.info("Saved %d rows to %s", len(issuer_rows), issuer_path)
    if rating_rows:
        pd.DataFrame(rating_rows).to_csv(ratings_path, index=False)
        logger.info("Saved %d rows to %s", len(rating_rows), ratings_path)


def collect(
    emitter_groups: dict[int, list[str]],
    cookies: list[dict[str, object]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Run the full collection pipeline, grouped by emitter.

    Opens a VISIBLE browser. On first launch, navigates to cbonds.ru
    so you can verify you're logged in before scraping starts.

    For each unique emitter_id, searches ONE representative ISIN via
    autocomplete to obtain the Cbonds company_id, then fetches issuer
    data once and fans out the results to all ISINs in the group.

    Returns (issuer_info_rows, rating_rows).
    """
    issuer_info_rows: list[dict[str, str]] = []
    rating_rows: list[dict[str, str]] = []

    emitters_found = 0
    emitters_not_found = 0
    total_isins_covered = 0

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

        page.goto("https://cbonds.ru/", timeout=60000)
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
        input(
            "Browser opened. Verify you're logged in to Cbonds, "
            "then press Enter here..."
        )

        emitter_items = list(emitter_groups.items())

        for idx, (emitter_id, isins) in enumerate(
            tqdm(emitter_items, desc="Emitters", unit="emitter")
        ):
            # --- Step 1: search for ONE representative ISIN from this group ---
            representative_isin = isins[0]
            bond_url = search_bond(page, representative_isin)
            polite_delay()

            if not bond_url:
                # Could not find this emitter on Cbonds -- fill empty rows
                # for all ISINs in the group
                emitters_not_found += 1
                for isin in isins:
                    issuer_info_rows.append({
                        "isin": isin,
                        "issuer_inn": "",
                        "sector": "",
                        "company_id": "",
                        "company_name": "",
                    })
                total_isins_covered += len(isins)
                continue

            emitters_found += 1

            # --- Step 2: scrape the bond page (extract company_id) ---
            bond_result = scrape_bond_page(page, representative_isin, bond_url)
            polite_delay()

            company_id = bond_result.company_id
            company_name = bond_result.company_name or ""

            # --- Step 3: fetch issuer data ONCE for this company_id ---
            issuer_result: IssuerResult | None = None
            if company_id:
                issuer_result = scrape_issuer(page, company_id)
                polite_delay()

            inn = issuer_result.inn if issuer_result else ""
            sector = issuer_result.sector if issuer_result else ""
            cname = issuer_result.company_name if issuer_result else company_name

            # --- Step 4: fan out results to ALL ISINs in the emitter group ---
            for isin in isins:
                issuer_info_rows.append({
                    "isin": isin,
                    "issuer_inn": inn,
                    "sector": sector,
                    "company_id": company_id or "",
                    "company_name": cname or company_name,
                })

                if issuer_result:
                    for entry in issuer_result.rating_history:
                        rating_rows.append({
                            "isin": isin,
                            "agency": entry["agency"],
                            "rating": entry["rating"],
                            "date_assigned": entry["date_assigned"],
                            "outlook": entry["outlook"],
                            "scale": entry["scale"],
                        })

            total_isins_covered += len(isins)

            if (idx + 1) % CHECKPOINT_INTERVAL == 0:
                logger.info(
                    "Checkpoint at emitter %d/%d (%d ISINs covered). "
                    "Saving intermediate results...",
                    idx + 1,
                    len(emitter_items),
                    total_isins_covered,
                )
                save_results(
                    issuer_info_rows,
                    rating_rows,
                    CHECKPOINT_ISSUER_INFO,
                    CHECKPOINT_RATINGS,
                )

        browser.close()

    logger.info("=" * 60)
    logger.info("COLLECTION SUMMARY")
    logger.info("=" * 60)
    logger.info("Total emitter groups:     %d", len(emitter_groups))
    logger.info("Emitters found on Cbonds: %d", emitters_found)
    logger.info("Emitters NOT found:       %d", emitters_not_found)
    logger.info("Total ISINs covered:      %d", total_isins_covered)
    logger.info("Total issuer_info rows:   %d", len(issuer_info_rows))
    logger.info("Total rating rows:        %d", len(rating_rows))
    logger.info("=" * 60)

    return issuer_info_rows, rating_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect credit ratings and issuer info from Cbonds.ru."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=f"Test mode: process only the first {TEST_MODE_LIMIT} emitters.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv(UNIVERSE_FILE)

    # Group ISINs by emitter_id -- one search per emitter instead of per ISIN
    emitter_groups: dict[int, list[str]] = (
        universe.dropna(subset=["emitter_id", "isin"])
        .groupby("emitter_id")["isin"]
        .apply(list)
        .to_dict()
    )
    total_isins = sum(len(v) for v in emitter_groups.values())
    logger.info(
        "Loaded %d unique emitters (%d ISINs) from universe.",
        len(emitter_groups),
        total_isins,
    )

    if args.test:
        # Keep only the first N emitter groups
        limited = dict(list(emitter_groups.items())[:TEST_MODE_LIMIT])
        test_isins = sum(len(v) for v in limited.values())
        emitter_groups = limited
        logger.info(
            "TEST MODE: processing only %d emitters (%d ISINs).",
            len(emitter_groups),
            test_isins,
        )

    cookies = load_cookies(COOKIES_FILE)

    issuer_info_rows, rating_rows = collect(emitter_groups, cookies)

    save_results(issuer_info_rows, rating_rows, OUTPUT_ISSUER_INFO, OUTPUT_RATINGS)

    for cp in (CHECKPOINT_ISSUER_INFO, CHECKPOINT_RATINGS):
        if cp.exists():
            cp.unlink()
            logger.info("Removed checkpoint: %s", cp)

    logger.info("Done. Output files:")
    logger.info("  %s", OUTPUT_ISSUER_INFO)
    logger.info("  %s", OUTPUT_RATINGS)


if __name__ == "__main__":
    main()
