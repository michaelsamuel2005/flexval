"""Price data for flexval.

Gas: Henry Hub daily spot, US$/MMBtu, from the U.S. Energy Information Administration via the `datasets/natural-gas`
data package (ODC-PDDL). Real prices. The series is sparse before 2008 and complete (about 250 trading days a year)
from 2008 to 14 October 2026, so calibration uses 2008 onwards.

Power: GB day-ahead prices from Ember's European wholesale electricity price data (CC-BY-4.0) when the file is present
in data/raw; otherwise a simulated series that is labelled SIMULATED everywhere it is used. The gas leg of the GB spark
spread is an NBP proxy built from Henry Hub dynamics at a stated GB price level; it is an assumption, not a price.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def henry_hub(start: str = "2008-01-01") -> pd.Series:
    d = pd.read_csv(RAW / "henry_hub_daily.csv", parse_dates=["Date"]).dropna()
    s = d.set_index("Date")["Price"].sort_index()
    s = s[s.index >= start]
    s.name = "henry_hub_usd_mmbtu"
    return s


def henry_hub_provenance() -> dict:
    p = json.loads((RAW / "natural-gas-datapackage.json").read_text())
    return {"source": "U.S. EIA via datasets/natural-gas", "licence": p["licenses"][0]["name"],
            "description": p["description"], "unit": "US$/MMBtu, nominal"}


def gb_power() -> tuple[pd.Series, dict]:
    """GB day-ahead daily average price, £/MWh. Real (Ember) if the file exists, else simulated and labelled."""
    files = sorted(glob.glob(str(RAW / "*wholesale*price*.csv")) + glob.glob(str(RAW / "*ember*.csv")))
    for f in files:
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        cols = {c.lower(): c for c in d.columns}
        country = next((cols[c] for c in cols if c in ("country", "iso3 code", "iso2 code", "country code")), None)
        date = next((cols[c] for c in cols if c in ("date", "datetime", "datetime (utc)", "datetime (local)")), None)
        price = next((cols[c] for c in cols if "price" in c), None)
        if not (country and date and price):
            continue
        gb = d[d[country].astype(str).str.lower().isin(["united kingdom", "great britain", "gbr", "gb", "uk"])]
        if gb.empty:
            continue
        s = pd.Series(gb[price].values, index=pd.to_datetime(gb[date], utc=True, errors="coerce").tz_localize(None))
        s = s.dropna().sort_index()
        s = s.resample("D").mean().dropna()
        s.name = "gb_day_ahead_gbp_mwh"
        return s, {"source": f"Ember European wholesale electricity price data ({Path(f).name})", "licence": "CC-BY-4.0",
                   "real": True, "unit": "£/MWh, daily average of day-ahead", "rows": int(len(s)),
                   "from": str(s.index.min().date()), "to": str(s.index.max().date())}
    return simulated_gb_power(), {"source": "SIMULATED (no Ember file in data/raw)", "licence": None, "real": False,
                                  "unit": "£/MWh, daily average of day-ahead", "note":
                                  "seasonal mean-reverting log-price with jumps; level and volatility chosen to resemble GB "
                                  "2023-2026 day-ahead statistics; not market data"}


def simulated_gb_power(seed: int = 11, start: str = "2016-01-01", end: str = "2026-10-14") -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="D")
    n = len(idx)
    t = np.arange(n) / 365.25
    season = 0.18 * np.cos(2 * np.pi * (t - 0.05)) + 0.04 * np.cos(4 * np.pi * t)
    level = np.log(75.0) + 0.25 * np.where(idx.year == 2022, 1.0, 0.0) + 0.12 * np.where(idx.year == 2021, 1.0, 0.0)
    x = np.zeros(n)
    kappa, sigma = 25.0 / 365.0, 0.085
    dow = np.where(idx.dayofweek >= 5, -0.10, 0.0)
    for i in range(1, n):
        jump = rng.normal(0.35, 0.15) if rng.random() < 0.004 else 0.0
        x[i] = x[i - 1] * (1 - kappa) + sigma * rng.standard_normal() + jump
    s = pd.Series(np.exp(level + season + dow + x), index=idx, name="gb_day_ahead_gbp_mwh")
    return s


def nbp_proxy(hh: pd.Series, level_gbp_mwh: float = 30.0) -> tuple[pd.Series, dict]:
    """Gas leg for GB spread deals: Henry Hub dynamics rescaled to a stated NBP level in £/MWh. An assumption."""
    s = hh / hh.mean() * level_gbp_mwh
    s.name = "nbp_proxy_gbp_mwh"
    return s, {"source": "NBP PROXY: Henry Hub daily series rescaled to a stated mean level", "real": False,
               "level_gbp_mwh": level_gbp_mwh, "note": "dynamics (seasonality, mean reversion, volatility, jumps) are Henry Hub's; "
               "the level is an assumption; replace with NBP/TTF settlements when available"}
