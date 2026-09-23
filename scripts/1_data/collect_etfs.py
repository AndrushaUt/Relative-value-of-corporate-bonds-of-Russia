"""Collect daily price history для БПИФов корп облигаций РФ.

Включаемые фонды (выбраны после market research, май 2026):
    - OBLG  : ВИМ — Российские корп облигации смарт-бета (был VTBB до 22.07.2022)
              tracks RUCBTRNS, TER 0.71%, ~1.88y duration
              ⇒ primary peer для нашей стратегии
    - SBRB  : Первая (Сбер) — Корп облигации, TER 0.76%, ~2.1y, IG (A-)
              ⇒ mainstream retail alternative
    - TBRU  : Т-Капитал Облигации (Тинькофф), TER 1.6%, ~2.33y, BBB-
              ⇒ актив но управляемый — прямой конкурент нашей active strategy

VTBB→OBLG сошиваем по дате 2022-07-22.

Output:
    data/etfs.csv: date, OBLG, SBRB, TBRU, *_ret columns

NOTE: Цены БПИФов уже включают купонный доход (total return basis),
поэтому daily return = price change только. NAV ETFs accrue купоны.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUT_PATH = DATA_DIR / "etfs.csv"

# Tickers map: канонический → (история, дата_переключения|None)
ETFS: dict[str, list[tuple[str, str, str]]] = {
    # canonical_ticker → [(secid, from_date, till_date), ...]
    "OBLG": [
        ("VTBB", "2021-01-01", "2022-07-21"),  # старый ticker
        ("OBLG", "2022-07-22", "2026-03-31"),  # новый ticker
    ],
    "SBRB": [
        ("SBRB", "2021-01-01", "2026-03-31"),
    ],
    "TBRU": [
        ("TBRU", "2021-01-01", "2026-03-31"),
    ],
}

BASE = "https://iss.moex.com/iss/history/engines/stock/markets/shares/securities"


def fetch_history(secid: str, date_from: str, date_till: str) -> pd.DataFrame:
    """Paginated fetch of daily history."""
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
                time.sleep(2 ** attempt)
                print(f"  retry {secid} (attempt {attempt + 1})", end=" ", flush=True)
        data = r.json().get("history", {})
        chunks = data.get("data", [])
        if not chunks:
            break
        for row in chunks:
            if row[1] is not None:
                rows.append({"date": row[0], "close": row[1]})
        start += len(chunks)
        if len(chunks) < 100:
            break
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"])
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def fetch_etf(canonical: str, segments: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Сшить multiple ticker segments в одну непрерывную серию по canonical."""
    parts = []
    for secid, dfrom, dtill in segments:
        print(f"  {canonical} ← {secid} {dfrom}..{dtill}...", end=" ", flush=True)
        df = fetch_history(secid, dfrom, dtill)
        print(f"{len(df)} rows")
        if not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    full = pd.concat(parts, ignore_index=True)
    full = full.drop_duplicates("date").sort_values("date").reset_index(drop=True)

    # Сшивка через ratio для смены ticker (VTBB → OBLG в июле 2022, lot size мог измениться)
    # Простой подход: normalise все цены, чтобы daily return был непрерывным.
    # Найти первый день нового tickerа и пересчитать его цену в continuous-price terms.
    if len(segments) > 1:
        # Split point — начало второго segments
        split_date = pd.Timestamp(segments[1][1])
        before = full[full["date"] < split_date]
        after = full[full["date"] >= split_date]
        if not before.empty and not after.empty:
            last_old = before["close"].iloc[-1]
            first_new = after["close"].iloc[0]
            ratio = last_old / first_new if first_new != 0 else 1.0
            full.loc[full["date"] >= split_date, "close"] *= ratio
            print(f"    {canonical}: rescaled new ticker by {ratio:.4f}")
    return full


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Fetching ETFs: {list(ETFS.keys())}")
    frames = []
    for canonical, segments in ETFS.items():
        df = fetch_etf(canonical, segments)
        if df.empty:
            print(f"  {canonical}: NO DATA")
            continue
        df = df.rename(columns={"close": canonical})
        print(f"  {canonical}: total {len(df)} rows ({df['date'].min().date()} .. {df['date'].max().date()})")
        frames.append(df)

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)

    for t in ETFS.keys():
        if t in merged.columns:
            merged[f"{t.lower()}_ret"] = merged[t].pct_change()

    merged.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")
    print(f"Shape: {merged.shape}")
    print(merged.tail(5))

    import numpy as np
    print("\nSanity Sharpe (full available range, no quality filter):")
    for t in ETFS.keys():
        col = f"{t.lower()}_ret"
        if col in merged.columns:
            r = merged[col].dropna().to_numpy()
            sh = np.sqrt(252) * r.mean() / r.std(ddof=1) if r.std() > 0 else float("nan")
            print(f"  {t}: Sharpe={sh:.2f}, n={len(r)}, mean_daily={r.mean()*100:.4f}%")


if __name__ == "__main__":
    main()
