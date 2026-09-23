"""Collect daily total-return index history from MOEX ISS for benchmarks.

Indices (актуальные на 2026):
    RUCBTRNS — Индекс МосБиржи Корпоративных Облигаций (Total Return New Standard).
               Заменяет RUCBITR с июня 2023. Покрывает 2021..present. Primary бенчмарк.
    RGBITR   — Russian Government Bond Total Return Index (OFZ). Secondary.

Output: data/benchmarks.csv with columns date, RUCBTRNS, RGBITR
        + daily returns rucbtrns_ret, rgbitr_ret.

Usage: python scripts/collect_benchmarks.py
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUT_PATH = DATA_DIR / "benchmarks.csv"

INDICES: list[str] = ["RUCBTRNS", "RGBITR"]
DATE_FROM = "2021-01-01"
DATE_TILL = "2026-03-31"

BASE = "https://iss.moex.com/iss/history/engines/stock/markets/index/securities"


def fetch_index(secid: str, date_from: str, date_till: str) -> pd.DataFrame:
    """Fetch daily close history for a MOEX index via ISS API.

    Returns DataFrame with columns date, close.
    """
    rows: list[dict] = []
    start = 0
    while True:
        params = {
            "from": date_from,
            "till": date_till,
            "start": start,
            "iss.meta": "off",
            "iss.only": "history",
            "history.columns": "TRADEDATE,CLOSE",
        }
        for attempt in range(4):
            try:
                r = requests.get(f"{BASE}/{secid}.json", params=params, timeout=60)
                r.raise_for_status()
                break
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt == 3:
                    raise
                backoff = 2 ** attempt
                print(f"  retry in {backoff}s ({e.__class__.__name__})...", end=" ", flush=True)
                time.sleep(backoff)
        data = r.json().get("history", {})
        chunks = data.get("data", [])
        if not chunks:
            break
        for row in chunks:
            rows.append({"date": row[0], "close": row[1]})
        start += len(chunks)
        if len(chunks) < 100:
            break
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"])
    return df.sort_values("date").reset_index(drop=True)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Fetching indices {INDICES} from {DATE_FROM} to {DATE_TILL}...")

    frames: list[pd.DataFrame] = []
    for secid in INDICES:
        print(f"  {secid}...", end=" ", flush=True)
        df = fetch_index(secid, DATE_FROM, DATE_TILL)
        df = df.rename(columns={"close": secid})
        print(f"{len(df)} rows ({df['date'].min().date()} .. {df['date'].max().date()})")
        frames.append(df)

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)

    # Daily returns (simple): r_t = close_t / close_{t-1} - 1
    for secid in INDICES:
        merged[f"{secid.lower()}_ret"] = merged[secid].pct_change()

    merged.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")
    print(f"Shape: {merged.shape}")
    print(merged.tail())


if __name__ == "__main__":
    main()
