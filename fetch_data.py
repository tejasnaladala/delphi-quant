"""Fetch S&P 500 daily bars from yfinance and cache to parquet.

Free, no API key. Uses the current S&P 500 constituent list. We accept
survivorship bias for the v0.1 backtester (acknowledged in PRE_REGISTRATION.md);
v0.2 will use the historical constituent list to remove the bias.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

START = "2010-01-01"
END = "2025-12-31"


def get_sp500_tickers() -> list[str]:
    """Hardcoded liquid US large-cap universe (top 100 by market cap, snapshot 2026-04).

    Survivorship-biased by construction (acknowledged in PRE_REGISTRATION.md).
    Hardcoded rather than scraped because Wikipedia blocks programmatic access
    and we want determinism across reruns.
    """
    return [
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA",
        "BRK-B", "JPM", "V", "UNH", "XOM", "WMT", "JNJ", "MA", "PG", "AVGO",
        "HD", "CVX", "ABBV", "MRK", "LLY", "PEP", "KO", "BAC", "PFE", "COST",
        "TMO", "CSCO", "MCD", "ACN", "ADBE", "ABT", "LIN", "CRM", "DHR", "AMD",
        "DIS", "WFC", "TXN", "NFLX", "VZ", "PM", "CMCSA", "RTX", "INTC", "NEE",
        "BMY", "QCOM", "ORCL", "T", "UPS", "HON", "COP", "LOW", "INTU", "AMGN",
        "UNP", "IBM", "MS", "SBUX", "NKE", "GS", "BA", "CAT", "AMT", "PLD",
        "DE", "BLK", "ELV", "AXP", "MDT", "GE", "ADP", "C", "MDLZ", "GILD",
        "TJX", "ISRG", "BKNG", "ADI", "MMC", "VRTX", "REGN", "LMT", "SYK",
        "MO", "PYPL", "ZTS", "SCHW", "CB", "CI", "SO", "PGR", "DUK", "BDX",
        "EOG", "AON", "BSX", "CL", "ITW", "ETN", "USB",
    ]


def fetch_universe(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch adjusted close + volume for the universe in batches."""
    all_data = []
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        print(f"  fetching {i + 1}-{min(i + batch_size, len(tickers))} of {len(tickers)}: {batch[:3]}...")
        try:
            df = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            all_data.append(df)
            time.sleep(0.5)
        except Exception as exc:
            print(f"    batch failed: {exc}")
    return pd.concat(all_data, axis=1) if all_data else pd.DataFrame()


def main() -> None:
    out_path = DATA_DIR / "sp500_daily.parquet"
    if out_path.exists():
        print(f"already cached at {out_path}, skipping fetch")
        return

    print("pulling current S&P 500 constituent list (survivorship-biased, v0.1 caveat)...")
    tickers = get_sp500_tickers()
    print(f"  {len(tickers)} tickers")

    print(f"fetching daily bars {START} to {END}...")
    df = fetch_universe(tickers, START, END)

    if df.empty:
        raise RuntimeError("no data fetched; check yfinance + network")

    print(f"saving to {out_path} (shape: {df.shape})...")
    df.to_parquet(out_path)
    print("done")


if __name__ == "__main__":
    main()
