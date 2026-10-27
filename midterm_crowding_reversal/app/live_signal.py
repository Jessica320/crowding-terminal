"""Today's crowding ranking from Binance's public API (shadow mode: observe only, never trade).

    python app/live_signal.py      # refresh outputs/live_cache.parquet the day before the demo

The rule score needs each coin's own 30-day funding history; the live z-score uses the
settled history in data/crypto/market/funding_rate.parquet plus the current estimate, so it
is labelled an estimate (the live rate is not settled until the next funding time).
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import OUT, SETUP  # noqa: E402

URL = "https://fapi.binance.com/fapi/v1/premiumIndex"
SYMS = SETUP["universe"]["symbols"]


def fetch_live(timeout: float = 10) -> pd.DataFrame:
    js = requests.get(URL, timeout=timeout).json()
    df = pd.DataFrame(js)
    df = df[df["symbol"].isin(SYMS)].copy()
    for c in ["markPrice", "indexPrice", "lastFundingRate"]:
        df[c] = df[c].astype(float)
    df["premium"] = df["markPrice"] / df["indexPrice"] - 1
    df["next_funding"] = pd.to_datetime(df["nextFundingTime"], unit="ms", utc=True)
    df["fetched_utc"] = datetime.now(timezone.utc)
    return df[["symbol", "lastFundingRate", "premium", "markPrice", "next_funding", "fetched_utc"]]


def rank(live: pd.DataFrame, funding_hist: pd.DataFrame | None) -> pd.DataFrame:
    out = live.copy()
    if funding_hist is not None and len(funding_hist):
        h = funding_hist.copy()
        h["day"] = h["timestamp"].dt.normalize()
        d = h.groupby(["symbol", "day"])["funding_rate"].sum().groupby("symbol").tail(30)
        st = d.groupby("symbol").agg(["mean", "std"])
        per_day = 24 / 8
        out = out.merge(st, left_on="symbol", right_index=True, how="left")
        out["funding_z_est"] = (out["lastFundingRate"] * per_day - out["mean"]) / out["std"]
        key = "funding_z_est"
    else:
        key = "lastFundingRate"
    out = out.sort_values(key, ascending=False).reset_index(drop=True)
    out["crowding_rank"] = range(1, len(out) + 1)
    out["shadow_action"] = "—"
    out.loc[out.index[:3], "shadow_action"] = "做空（最擠）"
    out.loc[out.index[-3:], "shadow_action"] = "做多（最不擠）"
    return out


if __name__ == "__main__":
    from src.config import RAW
    hist = pd.read_parquet(RAW["funding"]) if RAW["funding"].exists() else None
    r = rank(fetch_live(), hist)
    r.to_parquet(OUT / "live_cache.parquet", index=False)
    print(r[["crowding_rank", "symbol", "lastFundingRate", "premium", "shadow_action"]].to_string())
