"""Frozen setup, the holdout gatekeeper, and shared paths.

Research code reads data through ``research_only()``, which never returns a row on or after
``HOLDOUT_START``. Only ``08_holdout.py`` calls ``open_holdout_once()``, which writes a lock
file first, so a second opening is refused.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib
import pandas as pd
import yaml

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SETUP = yaml.safe_load((ROOT / "setup.yaml").read_text(encoding="utf-8"))

# CROWD_SANDBOX=<dir> redirects raw data, outputs, ledger and lock to <dir> so the code can be
# exercised on synthetic data without touching the real research record.
SANDBOX = Path(os.environ["CROWD_SANDBOX"]) if os.environ.get("CROWD_SANDBOX") else None
BASE = SANDBOX or ROOT
OUT = BASE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
PROCESSED = BASE / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)
LOG_PATH = BASE / "run_log.csv"

HOLDOUT_START = pd.Timestamp(SETUP["holdout"]["start"], tz="UTC")
HOLDOUT_END = pd.Timestamp(SETUP["holdout"]["end"], tz="UTC")
RESEARCH_START = pd.Timestamp("2020-01-01", tz="UTC")
LOCK = OUT / "HOLDOUT_OPENED.lock" if SANDBOX else REPO / SETUP["holdout"]["lock_file"]
RAW = {k: (SANDBOX / Path(v).name if SANDBOX else REPO / v) for k, v in SETUP["data"]["files"].items()}
COST_BPS = SETUP["costs"]["per_side_bps"]
PERIODS = 365  # crypto trades every calendar day

SIGNAL_FEATURES = SETUP["features"]["signal"]
STATE_FEATURES = SETUP["features"]["state"]
FEATURES = SIGNAL_FEATURES + STATE_FEATURES


def research_only(df: pd.DataFrame, ts_col: str = "date") -> pd.DataFrame:
    """Call first on anything read for research: drops every row in the holdout."""
    out = df[df[ts_col] < HOLDOUT_START]
    assert out[ts_col].max() < HOLDOUT_START, "holdout leaked!"
    return out


def open_holdout_once() -> str:
    if LOCK.exists():
        raise SystemExit(f"Holdout already opened at {LOCK.read_text()}; refusing a second look.")
    stamp = datetime.now().isoformat(timespec="seconds")
    LOCK.write_text(stamp, encoding="utf-8")
    return stamp


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def data_snapshot() -> str:
    h = hashlib.sha256()
    for p in RAW.values():
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def setup_hash() -> str:
    return hashlib.sha256((ROOT / "setup.yaml").read_bytes()).hexdigest()[:12]
