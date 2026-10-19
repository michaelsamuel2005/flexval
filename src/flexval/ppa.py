"""Fixed-price wind PPA: the buyer pays K per MWh for whatever the farm produces and sells it at the day-ahead price.
Value = sum of volume x (captured price - K), discounted. Two risks a fixed price does not remove: shape (wind sells
when prices are low: the capture rate is below 1) and volume (the farm's output is uncertain). Wind output is
SIMULATED: daily capacity factors from a seasonal Beta distribution with a stated cannibalisation parameter linking
high wind to lower prices. No real generation data is used, and the memo says so.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class PPADeal:
    name: str
    start: str
    days: int
    mw: float = 100.0
    strike: float = 60.0            # £/MWh fixed price paid to the generator
    mean_cf: float = 0.36           # annual mean capacity factor
    cf_seasonal_amp: float = 0.10   # winter-summer swing in capacity factor
    cannibalisation: float = 0.35   # price falls by this fraction x (cf - mean) relative to the market price
    r: float = 0.04

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start, periods=self.days, freq="D")

    def to_dict(self) -> dict:
        return asdict(self)


def simulate_wind(deal: PPADeal, n_paths: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = deal.dates
    doy = d.dayofyear.values
    mean = deal.mean_cf + deal.cf_seasonal_amp * np.cos(2 * np.pi * (doy - 15) / 365.25)
    mean = np.clip(mean, 0.05, 0.9)
    k = 4.0  # Beta concentration: daily variability
    a, b = mean * k, (1 - mean) * k
    return rng.beta(a[None, :], b[None, :], size=(n_paths, len(d)))


def value(deal: PPADeal, P: np.ndarray, cf: np.ndarray, df: np.ndarray) -> dict:
    """P: power price paths [n_paths, days]; cf: capacity-factor paths."""
    mean_cf = cf.mean()
    captured = P * (1 - deal.cannibalisation * (cf - mean_cf) / max(mean_cf, 1e-6))
    vol = cf * deal.mw * 24
    cash = vol * (captured - deal.strike) * df[None, :]
    per_path = cash.sum(1)
    baseload = (P * df[None, :]).mean(1)
    capture_rate = (captured * vol).sum(1) / np.maximum(vol.sum(1), 1e-9) / np.maximum((P).mean(1), 1e-9)
    annual_vol = vol.sum(1)
    return {"value": float(per_path.mean()), "se": float(per_path.std(ddof=1) / np.sqrt(len(per_path))), "per_path": per_path,
            "capture_rate": float(capture_rate.mean()), "annual_volume_mwh": float(annual_vol.mean()),
            "annual_volume_sd": float(annual_vol.std(ddof=1)), "breakeven_strike": float(
                ((captured * vol * df[None, :]).sum(1) / (vol * df[None, :]).sum(1)).mean())}
