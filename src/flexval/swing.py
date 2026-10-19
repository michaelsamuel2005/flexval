"""Swing (take-or-pay) gas contract: each month the buyer takes between q_min and q_max at a fixed price K, with an
annual minimum and maximum. Valued by least-squares Monte Carlo on the cumulative take (same scheme as storage) and
checked against the deterministic optimum on the forward curve when volatility is zero.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy.optimize import linprog


@dataclass
class SwingDeal:
    name: str
    start: str
    months: int
    strike: float            # fixed contract price, $/MMBtu
    q_min: float             # monthly minimum take, MMBtu
    q_max: float             # monthly maximum take
    annual_min: float
    annual_max: float
    r: float = 0.04
    steps: int = 4           # take grid per month: q_min + k * (q_max - q_min) / steps

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start, periods=self.months, freq="MS")

    def to_dict(self) -> dict:
        return asdict(self)


def discount(deal: SwingDeal, t0: pd.Timestamp) -> np.ndarray:
    tau = np.asarray((deal.dates - t0).days, dtype=float) / 365.25
    return np.exp(-deal.r * np.maximum(tau, 0))


def intrinsic_lp(deal: SwingDeal, F: np.ndarray, df: np.ndarray) -> dict:
    n = deal.months
    c = -df * (F - deal.strike)
    A = np.vstack([np.ones((1, n)), -np.ones((1, n))])
    b = np.array([deal.annual_max, -deal.annual_min])
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(deal.q_min, deal.q_max)] * n, method="highs")
    return {"value": float(-res.fun), "take": res.x}


def lsmc(deal: SwingDeal, paths: np.ndarray, df: np.ndarray, basis: int = 3) -> dict:
    n_paths, n = paths.shape
    dq = (deal.q_max - deal.q_min) / deal.steps
    takes = deal.q_min + dq * np.arange(deal.steps + 1)
    # cumulative-take grid in units of dq above n*q_min
    G = n * deal.steps + 1
    cum = np.arange(G) * dq                      # extra take above the minimum path
    base_total = n * deal.q_min
    ok_end = (base_total + cum >= deal.annual_min - 1e-9) & (base_total + cum <= deal.annual_max + 1e-9)
    V_next = np.where(ok_end, 0.0, -1e12)[None, :].repeat(n_paths, 0)
    for t in range(n - 1, -1, -1):
        S = paths[:, t]
        X = np.column_stack([S ** k for k in range(basis + 1)])
        cont = np.full((n_paths, G), -1e12)
        for g in range(G):
            y = V_next[:, g]
            if np.all(y <= -1e11):
                continue
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            cont[:, g] = X @ beta
        V_now = np.full((n_paths, G), -1e12)
        max_g = t * deal.steps                      # cumulative extra take reachable by month t
        for g in range(min(max_g, G - 1) + 1):
            best, best_cash, best_next = np.full(n_paths, -1e12), np.zeros(n_paths), np.full(n_paths, g)
            for k in range(deal.steps + 1):
                g2 = g + k
                if g2 >= G:
                    continue
                cash = df[t] * (S - deal.strike) * takes[k]
                total = cash + cont[:, g2]
                better = total > best
                best = np.where(better, total, best)
                best_cash = np.where(better, cash, best_cash)
                best_next = np.where(better, g2, best_next)
            V_now[:, g] = best_cash + V_next[np.arange(n_paths), best_next]
            V_now[best <= -1e11, g] = -1e12
        V_next = V_now
    vals = V_next[:, 0]
    return {"value": float(vals.mean()), "se": float(vals.std(ddof=1) / np.sqrt(n_paths)), "per_path": vals}
