"""Tolling agreement on a gas-fired plant: the right to convert gas into power at a fixed heat rate.

Daily granularity on day-ahead prices. Unconstrained, the deal is a strip of daily spark-spread options
max(P - HR * G - VOM - EF * carbon, 0) * MW * 24, priced in closed form (Kirk) on the model-implied forwards and by
Monte Carlo on correlated paths. With a start cost and a minimum run length the dispatch is a dynamic programme on
each path, and the difference between the two values is what the physical constraints cost.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .models import SpotModel, correlated_shocks, terminal_correlation
from .options import kirk, spread_mc


@dataclass
class TollingDeal:
    name: str
    start: str
    days: int
    mw: float = 400.0
    heat_rate: float = 1.90        # MWh gas per MWh power (about 53% efficiency, HHV)
    vom: float = 3.0               # £/MWh power
    emission_factor: float = 0.37  # tCO2 per MWh power
    carbon_price: float = 40.0     # £/tCO2, fixed input (no carbon price series available)
    start_cost: float = 15000.0    # £ per start
    min_run_days: int = 2
    r: float = 0.04

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start, periods=self.days, freq="D")

    @property
    def fixed_cost(self) -> float:
        return self.vom + self.emission_factor * self.carbon_price

    def to_dict(self) -> dict:
        return asdict(self)


def discount(deal: TollingDeal, t0: pd.Timestamp) -> np.ndarray:
    tau = np.asarray((deal.dates - t0).days, dtype=float) / 365.25
    return np.exp(-deal.r * np.maximum(tau, 0))


def strip_closed_form(deal: TollingDeal, power: SpotModel, gas: SpotModel, rho: float) -> dict:
    """Sum over days of Kirk's spread-option value on model-implied forwards and implied volatilities."""
    d = deal.dates
    Fp, Fg = power.forward(d, jumps=False), gas.forward(d, jumps=False)
    sp, sg = power.implied_vol(d), gas.implied_vol(d)
    tau = np.maximum(np.asarray((d - power.t0).days, dtype=float) / 365.25, 1e-6)
    rho_T = terminal_correlation(power, gas, rho, tau)
    vals, d1, d2 = [], [], []
    for i in range(len(d)):
        k = kirk(Fp[i], deal.heat_rate * Fg[i], deal.fixed_cost, tau[i], sp[i], sg[i], float(rho_T[i]), deal.r)
        vals.append(k.value)
        d1.append(k.greeks["delta_1"])
        d2.append(k.greeks["delta_2"])
    scale = deal.mw * 24
    return {"value": float(np.sum(vals) * scale), "per_day": np.array(vals) * scale,
            "delta_power_mwh": np.array(d1) * scale, "delta_gas_mwh": -np.array(d2) * scale * deal.heat_rate,
            "forward_power": Fp, "forward_gas": Fg, "terminal_rho_1y": float(rho_T[-1]), "intrinsic": float(np.sum(np.maximum(Fp - deal.heat_rate * Fg - deal.fixed_cost, 0)
                                                                              * np.exp(-deal.r * tau)) * scale)}


def simulate_pair(deal: TollingDeal, power: SpotModel, gas: SpotModel, rho: float, n_paths: int, seed: int,
                  jumps: bool = True):
    d = deal.dates
    z1, z2 = correlated_shocks(n_paths, len(d), rho, seed)
    P = power.simulate(d, n_paths, seed=seed, jumps=jumps, shocks=z1)
    G = gas.simulate(d, n_paths, seed=seed + 1, jumps=jumps, shocks=z2)
    return P, G


def strip_mc(deal: TollingDeal, P: np.ndarray, G: np.ndarray, df: np.ndarray) -> dict:
    spread = P - deal.heat_rate * G - deal.fixed_cost
    pay = np.maximum(spread, 0.0) * deal.mw * 24
    per_path = (pay * df[None, :]).sum(1)
    return {"value": float(per_path.mean()), "se": float(per_path.std(ddof=1) / np.sqrt(len(per_path))), "per_path": per_path,
            "run_days": float((spread > 0).mean() * deal.days)}


def dispatch_dp(deal: TollingDeal, P: np.ndarray, G: np.ndarray, df: np.ndarray) -> dict:
    """Perfect-foresight-per-path dispatch with a start cost and a minimum run: an upper bound on the constrained value
    that a daily dispatcher with the realised prices could reach. State = days already running (0 = off)."""
    n_paths, n = P.shape
    margin = (P - deal.heat_rate * G - deal.fixed_cost) * deal.mw * 24 * df[None, :]
    start = deal.start_cost
    R = deal.min_run_days
    # states: 0 off; 1..R running with k days completed (k>=R equals R)
    NEG = -1e15
    V = np.zeros((n_paths, R + 1))
    for t in range(n - 1, -1, -1):
        Vn = np.full((n_paths, R + 1), NEG)
        m = margin[:, t]
        # from off: stay off, or start (pay start cost, run today, move to state 1)
        stay = V[:, 0]
        go = -start * df[t] + m + V[:, min(1, R)]
        Vn[:, 0] = np.maximum(stay, go)
        for k in range(1, R + 1):
            cont = m + V[:, min(k + 1, R)]          # keep running
            if k >= R:
                Vn[:, k] = np.maximum(cont, V[:, 0])  # may switch off after the minimum run
            else:
                Vn[:, k] = cont                        # must keep running
        V = Vn
    per_path = V[:, 0]
    return {"value": float(per_path.mean()), "se": float(per_path.std(ddof=1) / np.sqrt(n_paths)), "per_path": per_path}
