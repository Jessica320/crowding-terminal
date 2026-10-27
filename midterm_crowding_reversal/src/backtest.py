"""Daily long-short backtest: long the top 3 scores, short the bottom 3, 01:00 open to 01:00 open.

Without overlays the position depends only on that day's scores, so the engine is
vectorised. The vol target and the drawdown ladder depend on the strategy's own past
returns, so ``apply_overlays`` runs a day-by-day loop that only reads t-1 information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SETUP

K = 3
MIN_NAMES = 8
CAP = SETUP["constraints"]["max_weight_per_name"]


def _cap_side(w: pd.Series, total: float) -> pd.Series:
    """Scale to ``total`` with no name above CAP (water-filling)."""
    w = w / w.sum() * total
    for _ in range(10):
        over = w > CAP + 1e-12
        if not over.any():
            break
        excess = (w[over] - CAP).sum()
        w[over] = CAP
        free = ~over
        w[free] += excess * w[free] / w[free].sum()
    return w


def day_weights(day: pd.DataFrame, score: str, allocator: str = "equal") -> pd.Series:
    d = day[day["tradable"] & day[score].notna()].sort_values(score)
    if len(d) < MIN_NAMES:
        return pd.Series(dtype=float)
    longs, shorts = d.iloc[-K:], d.iloc[:K]
    if allocator == "inv_vol":
        wl = 1 / longs.set_index("symbol")["rvol_20d"].clip(lower=1e-4)
        ws = 1 / shorts.set_index("symbol")["rvol_20d"].clip(lower=1e-4)
    else:
        wl = pd.Series(1.0, index=longs["symbol"])
        ws = pd.Series(1.0, index=shorts["symbol"])
    return pd.concat([_cap_side(wl, 0.5), -_cap_side(ws, 0.5)])


def weights(panel: pd.DataFrame, score: str, allocator: str = "equal") -> pd.DataFrame:
    W = {dt: day_weights(g, score, allocator) for dt, g in panel.groupby("date")}
    return pd.DataFrame(W).T.sort_index().fillna(0.0)


def run(panel: pd.DataFrame, score: str, cost_bps: float = 7.0, allocator: str = "equal",
        W: pd.DataFrame | None = None) -> pd.DataFrame:
    if W is None:
        W = weights(panel, score, allocator)
    R = panel.pivot(index="date", columns="symbol", values="ret_long_simple").reindex(index=W.index, columns=W.columns)
    P = panel.pivot(index="date", columns="symbol", values="ret_price").reindex(index=W.index, columns=W.columns)
    F = panel.pivot(index="date", columns="symbol", values="funding_hold").reindex(index=W.index, columns=W.columns)
    gross = (W * R.fillna(0)).sum(axis=1)
    # weights drift with price over the 24h; the trade is target minus drifted holding
    drift = (W.shift(1) * (1 + P.shift(1).fillna(0)))
    drift = drift.div((1 + (W.shift(1) * P.shift(1).fillna(0)).sum(axis=1)), axis=0).fillna(0)
    turnover = (W - drift).abs().sum(axis=1)
    turnover.iloc[0] = W.iloc[0].abs().sum()
    out = pd.DataFrame({"gross": gross, "turnover": turnover,
                        "funding_pnl": -(W * F.fillna(0)).sum(axis=1),
                        "n_names": (W != 0).sum(axis=1)})
    out["net"] = out["gross"] - out["turnover"] * cost_bps / 1e4
    return out


def breakeven_bps(bt: pd.DataFrame) -> float:
    return float(bt["gross"].sum() / bt["turnover"].sum() * 1e4) if bt["turnover"].sum() else np.nan


def apply_overlays(bt: pd.DataFrame, cost_bps: float = 7.0, vol_target: float | None = 0.20,
                   ladder: dict | None = None) -> pd.DataFrame:
    """Scale each day's book using only returns through t-1.

    Vol target: scale = min(1, target / realised 20d vol of the base net return).
    Ladder (on the base strategy's drawdown, so it can recover while flat):
    dd >= halve -> 0.5x, dd >= stop -> 0x.  ``warn`` only flags.
    """
    ladder = ladder or SETUP["portfolio_extras"]["drawdown_ladder"]
    base = bt["gross"] - bt["turnover"] * cost_bps / 1e4
    scale = np.ones(len(bt))
    eq, peak = 1.0, 1.0
    vols = base.rolling(20, min_periods=10).std().shift(1) * np.sqrt(365)
    dd_hist = np.zeros(len(bt))
    for i in range(len(bt)):
        dd = eq / peak - 1
        dd_hist[i] = dd
        s = 1.0
        if vol_target is not None and not np.isnan(vols.iloc[i]) and vols.iloc[i] > 0:
            s = min(1.0, vol_target / vols.iloc[i])
        if ladder:
            if dd <= -ladder["stop"]:
                s = 0.0
            elif dd <= -ladder["halve"]:
                s *= 0.5
        scale[i] = s
        eq *= 1 + base.iloc[i]
        peak = max(peak, eq)
    out = bt.copy()
    out["scale"] = scale
    out["base_dd"] = dd_hist
    sc = pd.Series(scale, index=bt.index)
    # changing the scale is itself a trade on the whole book
    extra = (sc.diff().abs().fillna(0)) * (bt["n_names"] > 0)
    out["net"] = sc * bt["gross"] - (sc * bt["turnover"] + extra) * cost_bps / 1e4
    return out
