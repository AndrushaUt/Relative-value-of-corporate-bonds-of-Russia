#!/usr/bin/env python3
"""
Collect macroeconomic data from CBR API and MOEX ISS API.

Sources:
  A) CBR Key Rate (SOAP via DailyInfoWebServ)
  B) USD/RUB exchange rate (CBR XML)
  C) IMOEX index close (MOEX ISS JSON)
  D) RVI -- Russian Volatility Index (MOEX ISS JSON)
  E) Brent oil price via MOEX nearest-month futures (MOEX ISS JSON)

Output: data/macro.csv with columns:
  date, key_rate, usdrub, imoex_close, imoex_return, imoex_realized_vol,
  rvi, oil_brent
"""

import argparse
import os
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

DATA_DIR = str(Path(__file__).resolve().parents[2] / "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "macro.csv")

DATE_START = "2019-01-01"
DATE_END = "2026-03-30"

# CBR API expects DD.MM.YYYY -- derive from DATE_START / DATE_END
_CBR_START = datetime.strptime(DATE_START, "%Y-%m-%d").strftime("%d.%m.%Y")
_CBR_END = datetime.strptime(DATE_END, "%Y-%m-%d").strftime("%d.%m.%Y")

BOUNDS = {
    "key_rate": (0, 30),
    "usdrub": (30, 200),
    "imoex_close": (500, 10_000),
    "rvi": (5, 150),
    "oil_brent": (10, 200),
}

MAX_RETRIES = 5
BACKOFF_BASE = 2.0  # seconds
REQUEST_TIMEOUT = 30  # seconds

# CBR SOAP endpoint for key rate
_CBR_SOAP_URL = "https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx"
_CBR_SOAP_ACTION = "http://web.cbr.ru/KeyRate"

_KEY_RATE_SOAP_TEMPLATE = """\
<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/"
               xmlns:web="http://web.cbr.ru/">
  <soap:Body>
    <web:KeyRate>
      <web:fromDate>{from_date}</web:fromDate>
      <web:ToDate>{to_date}</web:ToDate>
    </web:KeyRate>
  </soap:Body>
</soap:Envelope>"""

# Brent futures month codes
_MONTH_CODES = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}


def _make_session() -> requests.Session:
    """Create a session with connection pooling and sensible headers."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (kursach-macro-collector/1.0)",
        "Accept-Encoding": "gzip, deflate",
    })
    return s


SESSION = _make_session()


def _request_with_retries(url: str, params: dict | None = None) -> requests.Response:
    """GET with exponential backoff. Raises on final failure."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp
        except (requests.RequestException, requests.HTTPError) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                wait = BACKOFF_BASE ** attempt
                print(f"  [retry {attempt}/{MAX_RETRIES}] {exc} -- waiting {wait:.0f}s")
                time.sleep(wait)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {last_exc}") from last_exc


def _post_with_retries(
    url: str,
    data: str | bytes,
    headers: dict[str, str],
) -> requests.Response:
    """POST with exponential backoff. Raises on final failure."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = SESSION.post(
                url, data=data, headers=headers, timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            return resp
        except (requests.RequestException, requests.HTTPError) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                wait = BACKOFF_BASE ** attempt
                print(f"  [retry {attempt}/{MAX_RETRIES}] {exc} -- waiting {wait:.0f}s")
                time.sleep(wait)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {last_exc}") from last_exc


def fetch_xml(url: str, params: dict | None = None) -> ET.Element:
    """Fetch URL and return parsed XML root element."""
    resp = _request_with_retries(url, params)
    return ET.fromstring(resp.content)


def fetch_json(url: str, params: dict | None = None) -> dict:
    """Fetch URL and return parsed JSON."""
    resp = _request_with_retries(url, params)
    return resp.json()


# A) CBR Key Rate (SOAP)

def collect_key_rate(test: bool = False) -> pd.DataFrame:
    """
    Fetch key rate from CBR SOAP service (DailyInfoWebServ/KeyRate).
    Returns DataFrame with columns: [date, key_rate].
    The SOAP endpoint returns daily values, so forward-fill is only
    needed for weekends/holidays.
    """
    print("\n--- Collecting CBR Key Rate ---")

    if test:
        # Start from 2018 to capture Dec-2018 rate change
        from_date, to_date = "2018-01-01", "2019-01-31"
    else:
        from_date = DATE_START
        to_date = DATE_END

    soap_body = _KEY_RATE_SOAP_TEMPLATE.format(
        from_date=from_date, to_date=to_date,
    )
    headers = {
        "Content-Type": "text/xml; charset=utf-8",
        "SOAPAction": _CBR_SOAP_ACTION,
    }

    resp = _post_with_retries(_CBR_SOAP_URL, data=soap_body.encode("utf-8"), headers=headers)
    root = ET.fromstring(resp.content)

    rows: list[dict[str, object]] = []
    for kr in root.iter("KR"):
        # Guard against missing elements / None text
        dt_elem = kr.find("DT")
        rate_elem = kr.find("Rate")
        if dt_elem is None or dt_elem.text is None:
            warnings.warn("Skipping KR record: missing <DT> element or text")
            continue
        if rate_elem is None or rate_elem.text is None:
            warnings.warn("Skipping KR record: missing <Rate> element or text")
            continue

        # DT is ISO-like: "2019-01-31T00:00:00+03:00" -- take date part
        dt_text = dt_elem.text.strip().split("T")[0]  # YYYY-MM-DD
        rate_text = rate_elem.text.strip().replace(",", ".")
        dt = datetime.strptime(dt_text, "%Y-%m-%d").date()
        rate = float(rate_text)
        rows.append({"date": dt, "key_rate": rate})

    df = pd.DataFrame(rows)
    if df.empty:
        print("  WARNING: no key rate records returned")
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
    print(f"  Fetched {len(df)} key rate daily records")
    return df


# B) USD/RUB exchange rate

def collect_usdrub(test: bool = False) -> pd.DataFrame:
    """
    Fetch USD/RUB official rate from CBR, split into yearly chunks.
    Returns DataFrame with columns: [date, usdrub].
    """
    print("\n--- Collecting USD/RUB ---")

    url = "https://www.cbr.ru/scripts/XML_dynamic.asp"
    val_code = "R01235"  # USD

    start_year = datetime.strptime(DATE_START, "%Y-%m-%d").year
    end_year = datetime.strptime(DATE_END, "%Y-%m-%d").year

    if test:
        year_ranges = [(start_year, start_year)]
    else:
        year_ranges = [(y, y) for y in range(start_year, end_year + 1)]

    all_rows = []
    for y_start, y_end in tqdm(year_ranges, desc="  USD/RUB years"):
        d1 = f"01.01.{y_start}"
        d2 = f"31.12.{y_end}" if y_end < end_year else _CBR_END
        params = {
            "date_req1": d1,
            "date_req2": d2,
            "VAL_NM_RQ": val_code,
        }
        root = fetch_xml(url, params)
        for rec in root.iter("Record"):
            dt_text = rec.attrib.get("Date", "").strip()       # DD.MM.YYYY
            # BUG 5 fix: skip records with empty Date attribute
            if not dt_text:
                warnings.warn("Skipping Record with empty Date attribute")
                continue
            val_elem = rec.find("Value")
            if val_elem is None or val_elem.text is None:
                warnings.warn("Skipping Record: missing <Value> element or text")
                continue
            val_text = val_elem.text.strip()
            val_text = val_text.replace(",", ".")
            dt = datetime.strptime(dt_text, "%d.%m.%Y").date()
            value = float(val_text)
            all_rows.append({"date": dt, "usdrub": value})
        # small pause between yearly chunks
        time.sleep(0.3)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("  WARNING: no USD/RUB records returned")
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
    print(f"  Fetched {len(df)} USD/RUB records")
    return df


# C) IMOEX Index Close

def collect_imoex(test: bool = False) -> pd.DataFrame:
    """
    Fetch IMOEX daily close from MOEX ISS with pagination.
    Returns DataFrame with columns: [date, imoex_close].
    """
    print("\n--- Collecting IMOEX Index ---")

    url = (
        "https://iss.moex.com/iss/history/engines/stock/markets/index"
        "/boards/SNDX/securities/IMOEX.json"
    )

    all_rows = []
    start = 0
    page_size = 100

    pbar = tqdm(desc="  IMOEX pages", unit="page")
    while True:
        params = {
            "from": DATE_START,
            "till": DATE_END,
            "iss.meta": "off",
            "iss.only": "history",
            "start": start,
        }
        data = fetch_json(url, params)

        history = data.get("history", {})
        columns = history.get("columns", [])
        rows = history.get("data", [])

        if not rows:
            break

        try:
            idx_date = columns.index("TRADEDATE")
            idx_close = columns.index("CLOSE")
        except ValueError as exc:
            print(f"  WARNING: expected columns not found: {exc}")
            print(f"  Available columns: {columns}")
            break

        for row in rows:
            trade_date = row[idx_date]
            close_val = row[idx_close]
            if close_val is not None:
                all_rows.append({
                    "date": trade_date,
                    "imoex_close": float(close_val),
                })

        pbar.update(1)
        # BUG 6 fix: advance by actual rows returned, not assumed page_size
        start += len(rows)

        if test:
            break  # only first page in test mode

        time.sleep(0.1)

    pbar.close()

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("  WARNING: no IMOEX records returned")
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
    print(f"  Fetched {len(df)} IMOEX records")
    return df


# D) RVI -- Russian Volatility Index

def collect_rvi(test: bool = False) -> pd.DataFrame:
    """
    Fetch RVI (Russian Volatility Index) daily close from MOEX ISS with pagination.
    Returns DataFrame with columns: [date, rvi].
    """
    print("\n--- Collecting RVI ---")

    url = (
        "https://iss.moex.com/iss/history/engines/stock/markets/index"
        "/securities/RVI.json"
    )

    all_rows: list[dict[str, object]] = []
    start = 0

    pbar = tqdm(desc="  RVI pages", unit="page")
    while True:
        params = {
            "from": DATE_START,
            "till": DATE_END,
            "iss.meta": "off",
            "iss.only": "history",
            "history.columns": "TRADEDATE,CLOSE",
            "start": start,
        }
        data = fetch_json(url, params)

        history = data.get("history", {})
        columns = history.get("columns", [])
        rows = history.get("data", [])

        if not rows:
            break

        try:
            idx_date = columns.index("TRADEDATE")
            idx_close = columns.index("CLOSE")
        except ValueError as exc:
            print(f"  WARNING: expected columns not found: {exc}")
            print(f"  Available columns: {columns}")
            break

        for row in rows:
            trade_date = row[idx_date]
            close_val = row[idx_close]
            if close_val is not None:
                all_rows.append({
                    "date": trade_date,
                    "rvi": float(close_val),
                })

        pbar.update(1)
        start += len(rows)

        if test:
            break  # only first page in test mode

        time.sleep(0.1)

    pbar.close()

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("  WARNING: no RVI records returned")
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
    print(f"  Fetched {len(df)} RVI records")
    return df


# E) Brent oil price via MOEX nearest-month futures

def collect_brent(test: bool = False) -> pd.DataFrame:
    """
    Fetch Brent oil price from MOEX futures (nearest-month contract).

    For each calendar month in the period, constructs the ticker for the
    nearest-month future (e.g. BRJ4 = April 2024) and fetches CLOSE prices.
    Concatenates all months and deduplicates by TRADEDATE.

    Returns DataFrame with columns: [date, oil_brent].
    """
    print("\n--- Collecting Brent Oil (MOEX futures) ---")

    base_url = (
        "https://iss.moex.com/iss/history/engines/futures/markets/forts"
        "/boards/RFUD/securities"
    )

    start_dt = datetime.strptime(DATE_START, "%Y-%m-%d")
    end_dt = datetime.strptime(DATE_END, "%Y-%m-%d")

    # Build list of (year, month) pairs covering the period
    if test:
        # Test mode: only January 2020
        month_list = [(2020, 1)]
    else:
        month_list: list[tuple[int, int]] = []
        y, m = start_dt.year, start_dt.month
        while (y, m) <= (end_dt.year, end_dt.month):
            month_list.append((y, m))
            m += 1
            if m > 12:
                m = 1
                y += 1

    all_rows: list[dict[str, object]] = []

    for year, month in tqdm(month_list, desc="  Brent months"):
        month_code = _MONTH_CODES[month]
        year_suffix = year % 10  # MOEX uses single digit: BRG0 = Feb 2020
        ticker = f"BR{month_code}{year_suffix}"
        url = f"{base_url}/{ticker}.json"

        from_date = f"{year}-{month:02d}-01"
        till_date = f"{year}-{month:02d}-28"

        start = 0
        while True:
            params = {
                "from": from_date,
                "till": till_date,
                "iss.meta": "off",
                "iss.only": "history",
                "history.columns": "TRADEDATE,SECID,CLOSE",
                "start": start,
            }
            try:
                data = fetch_json(url, params)
            except RuntimeError:
                # Some tickers may not exist (e.g. future months); skip
                print(f"  WARNING: could not fetch {ticker}, skipping")
                break

            history = data.get("history", {})
            columns = history.get("columns", [])
            rows = history.get("data", [])

            if not rows:
                break

            try:
                idx_date = columns.index("TRADEDATE")
                idx_close = columns.index("CLOSE")
            except ValueError as exc:
                print(f"  WARNING: expected columns not found for {ticker}: {exc}")
                break

            for row in rows:
                trade_date = row[idx_date]
                close_val = row[idx_close]
                if close_val is not None:
                    all_rows.append({
                        "date": trade_date,
                        "oil_brent": float(close_val),
                    })

            start += len(rows)

            if test:
                break  # only first page per month in test mode

            time.sleep(0.1)

        time.sleep(0.1)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("  WARNING: no Brent records returned")
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date", keep="first")
    print(f"  Fetched {len(df)} Brent records")
    return df


def sanity_check(df: pd.DataFrame) -> None:
    """Print warnings for values outside expected bounds."""
    print("\n--- Sanity Checks ---")
    ok = True
    for col, (lo, hi) in BOUNDS.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        out_of_range = series[(series < lo) | (series > hi)]
        if not out_of_range.empty:
            ok = False
            print(f"  WARNING: {col} has {len(out_of_range)} values outside [{lo}, {hi}]")
            print(f"    min={series.min():.4f}  max={series.max():.4f}")
        else:
            print(f"  OK: {col} in [{lo}, {hi}]  (min={series.min():.4f}, max={series.max():.4f})")
    if ok:
        print("  All sanity checks passed.")


def merge_macro(
    key_rate_df: pd.DataFrame | None,
    usdrub_df: pd.DataFrame | None,
    imoex_df: pd.DataFrame | None,
    rvi_df: pd.DataFrame | None,
    brent_df: pd.DataFrame | None,
    test: bool = False,
) -> pd.DataFrame:
    """
    Build daily index and merge all sources with forward-fill.
    Returns the final macro DataFrame.
    """
    print("\n--- Merging into daily panel ---")

    # 1. Daily date index
    if test:
        end = "2019-01-31"
    else:
        end = DATE_END
    daily = pd.DataFrame({
        "date": pd.date_range(DATE_START, end, freq="D"),
    })

    # 2. Key rate: merge and forward-fill (SOAP returns daily values,
    #    but weekends/holidays still need forward-fill)
    if key_rate_df is not None and not key_rate_df.empty:
        daily = daily.merge(key_rate_df[["date", "key_rate"]], on="date", how="left")
        daily["key_rate"] = daily["key_rate"].ffill()
    else:
        daily["key_rate"] = pd.NA

    # 3. USD/RUB: forward-fill for weekends/holidays
    if usdrub_df is not None and not usdrub_df.empty:
        daily = daily.merge(usdrub_df[["date", "usdrub"]], on="date", how="left")
        daily["usdrub"] = daily["usdrub"].ffill()
    else:
        daily["usdrub"] = pd.NA

    # 4. IMOEX: compute return BEFORE forward-fill (BUG 3 fix)
    #    imoex_return should be NaN on non-trading days.
    if imoex_df is not None and not imoex_df.empty:
        imoex = imoex_df[["date", "imoex_close"]].copy()
        imoex = imoex.sort_values("date")
        imoex["imoex_return"] = imoex["imoex_close"].pct_change() * 100

        # Merge raw return (NaN on non-trading days -- correct)
        daily = daily.merge(imoex[["date", "imoex_return"]], on="date", how="left")
        # Merge close and forward-fill for non-trading days
        daily = daily.merge(imoex[["date", "imoex_close"]], on="date", how="left")
        daily["imoex_close"] = daily["imoex_close"].ffill()

        # Rolling 20-day realized volatility of imoex_return
        # Computed on the trading-day returns already merged (NaN on non-trading days)
        daily["imoex_realized_vol"] = daily["imoex_return"].rolling(window=20, min_periods=15).std()
    else:
        daily["imoex_close"] = pd.NA
        daily["imoex_return"] = pd.NA
        daily["imoex_realized_vol"] = pd.NA

    # 5. RVI: forward-fill for non-trading days
    if rvi_df is not None and not rvi_df.empty:
        daily = daily.merge(rvi_df[["date", "rvi"]], on="date", how="left")
        daily["rvi"] = daily["rvi"].ffill()
    else:
        daily["rvi"] = pd.NA

    # 6. Brent oil: forward-fill for non-trading days
    if brent_df is not None and not brent_df.empty:
        daily = daily.merge(brent_df[["date", "oil_brent"]], on="date", how="left")
        daily["oil_brent"] = daily["oil_brent"].ffill()
    else:
        daily["oil_brent"] = pd.NA

    output_cols = [
        "date", "key_rate", "usdrub", "imoex_close", "imoex_return",
        "imoex_realized_vol", "rvi", "oil_brent",
    ]
    daily = daily[[c for c in output_cols if c in daily.columns]]

    print(f"  Daily panel: {len(daily)} rows, {daily.columns.tolist()}")
    return daily


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect macroeconomic data (CBR key rate, USD/RUB, IMOEX, RVI, Brent)."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Test mode: key_rate Jan 2019 only, USD/RUB 2019 only, "
            "IMOEX first page only, RVI first page only, Brent Jan 2020 only."
        ),
    )
    args = parser.parse_args()

    if args.test:
        print("=== TEST MODE ===")

    os.makedirs(DATA_DIR, exist_ok=True)

    # Collect each source independently; failures produce None
    key_rate_df = None
    usdrub_df = None
    imoex_df = None
    rvi_df = None
    brent_df = None

    try:
        key_rate_df = collect_key_rate(test=args.test)
    except Exception as exc:
        print(f"\n  ERROR collecting key rate: {exc}")

    try:
        usdrub_df = collect_usdrub(test=args.test)
    except Exception as exc:
        print(f"\n  ERROR collecting USD/RUB: {exc}")

    try:
        imoex_df = collect_imoex(test=args.test)
    except Exception as exc:
        print(f"\n  ERROR collecting IMOEX: {exc}")

    try:
        rvi_df = collect_rvi(test=args.test)
    except Exception as exc:
        print(f"\n  ERROR collecting RVI: {exc}")

    try:
        brent_df = collect_brent(test=args.test)
    except Exception as exc:
        print(f"\n  ERROR collecting Brent: {exc}")

    macro = merge_macro(
        key_rate_df, usdrub_df, imoex_df, rvi_df, brent_df, test=args.test,
    )

    sanity_check(macro)

    macro.to_csv(OUTPUT_PATH, index=False)
    print(f"\n--- Saved to {OUTPUT_PATH} ---")

    print("\n--- Summary ---")
    non_null = {col: macro[col].notna().sum() for col in macro.columns if col != "date"}
    print(f"  Date range: {macro['date'].min().date()} to {macro['date'].max().date()}")
    print(f"  Total rows: {len(macro)}")
    print(f"  Non-null counts: {non_null}")
    print(f"\n  Last 5 rows:")
    print(macro.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
