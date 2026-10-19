"""Gas storage contract: intrinsic value by linear programming, rolling intrinsic as a spot dispatch policy on simulated
paths, and extrinsic value by least-squares Monte Carlo (Boogert and de Jong, 2008) and by dynamic programming on a
price-inventory lattice (Tauchen discretisation of the OU deviation). Monthly decisions.

With volatility set to zero the paths collapse onto the forward curve and LSMC and the lattice must reproduce the LP
intrinsic value: that is the scorecard's deterministic check.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import norm

from .models import SpotModel


@dataclass
class StorageDeal:
    name: str
    start: str                 # first delivery month (YYYY-MM-01)
    months: int                # number of monthly periods
    capacity: float            # working gas, MMBtu
    inj_rate: float            # max injection per month, MMBtu
    wd_rate: float             # max withdrawal per month, MMBtu
    inj_cost: float = 0.02     # $/MMBtu
    wd_cost: float = 0.02      # $/MMBtu
    fuel_loss: float = 0.01    # fuel in kind: injecting 1 unit costs (1 + fuel_loss) units at spot
    start_inventory: float = 0.0
    end_inventory: float = 0.0
    r: float = 0.04            # continuously compounded discount rate
    grid: int = 10             # inventory grid steps (capacity / grid must divide the rates)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start, periods=self.months, freq="MS")

    def to_dict(self) -> dict:
        return asdict(self)


def discount_factors(deal: StorageDeal, t0: pd.Timestamp) -> np.ndarray:
    tau = np.asarray((deal.dates - t0).days, dtype=float) / 365.25
    return np.exp(-deal.r * np.maximum(tau, 0))


# ---- intrinsic: linear programme -------------------------------------------------------------------------------------
def intrinsic_lp(deal: StorageDeal, F: np.ndarray, df: np.ndarray, start_inventory: float | None = None, t_from: int = 0):
    """Maximise discounted cash flow over months t_from..N-1 given forward prices F (same length as deal.dates)."""
    n = deal.months - t_from
    I0 = deal.start_inventory if start_inventory is None else start_inventory
    Fs, dfs = F[t_from:], df[t_from:]
    # variables: inj_0..inj_{n-1}, wd_0..wd_{n-1}
    c = np.concatenate([dfs * (Fs * (1 + deal.fuel_loss) + deal.inj_cost), -dfs * (Fs - deal.wd_cost)])  # minimise -value
    L = np.tril(np.ones((n, n)))
    A_ub = np.vstack([np.hstack([L, -L]), np.hstack([-L, L])])
    b_ub = np.concatenate([np.full(n, deal.capacity - I0), np.full(n, I0)])
    A_eq = np.hstack([np.ones((1, n)), -np.ones((1, n))])
    b_eq = np.array([deal.end_inventory - I0])
    bounds = [(0, deal.inj_rate)] * n + [(0, deal.wd_rate)] * n
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        return {"value": -np.inf, "inj": np.zeros(n), "wd": np.zeros(n), "status": res.message}
    inj, wd = res.x[:n], res.x[n:]
    inv = I0 + np.cumsum(inj - wd)
    return {"value": float(-res.fun), "inj": inj, "wd": wd, "inventory": inv, "status": "optimal"}


# ---- rolling intrinsic on simulated paths -------------------------------------------------------------------------
def rolling_intrinsic(deal: StorageDeal, model: SpotModel, paths: np.ndarray, x_paths: np.ndarray, df: np.ndarray):
    """At each month re-solve the LP on the forward curve implied by the path's state and act on the first month at spot.
    paths, x_paths: [n_paths, months] spot prices and OU deviations on the delivery months."""
    n_paths, n = paths.shape
    dates = deal.dates
    values = np.zeros(n_paths)
    tau = np.asarray((dates - model.t0).days, dtype=float) / 365.25
    seas = model.seasonal(dates)
    for p in range(n_paths):
        inv = deal.start_inventory
        for t in range(n):
            # forward curve seen at month t from state x_t
            dt_ = np.maximum(tau[t:] - tau[t], 0)
            m = x_paths[p, t] * np.exp(-model.kappa * dt_)
            v = model.sigma ** 2 * (1 - np.exp(-2 * model.kappa * dt_)) / (2 * model.kappa)
            Ft = np.exp(seas[t:] + m + 0.5 * v)
            Ft[0] = paths[p, t]
            sol = intrinsic_lp(deal, np.concatenate([np.zeros(t), Ft]), df, start_inventory=inv, t_from=t)
            if sol["value"] == -np.inf:
                break
            a_i, a_w = sol["inj"][0], sol["wd"][0]
            cash = -(paths[p, t] * (1 + deal.fuel_loss) + deal.inj_cost) * a_i + (paths[p, t] - deal.wd_cost) * a_w
            values[p] += df[t] * cash
            inv = inv + a_i - a_w
    return {"value": float(values.mean()), "se": float(values.std(ddof=1) / np.sqrt(n_paths)), "per_path": values}


# ---- least-squares Monte Carlo --------------------------------------------------------------------------------------
def _actions(deal: StorageDeal):
    step = deal.capacity / deal.grid
    k_i, k_w = int(round(deal.inj_rate / step)), int(round(deal.wd_rate / step))
    return step, k_i, k_w


def lsmc(deal: StorageDeal, paths: np.ndarray, df: np.ndarray, basis: int = 3, seed: int = 0):
    """Boogert-de Jong LSMC on monthly spot paths [n_paths, months]. Inventory on a grid of `deal.grid` steps.
    Returns the value (mean over paths at the start inventory) and its standard error."""
    n_paths, n = paths.shape
    step, k_i, k_w = _actions(deal)
    G = deal.grid + 1
    # terminal penalty: inventory must end at end_inventory (within one grid step); otherwise large penalty
    end_k = int(round(deal.end_inventory / step))
    V_next = np.where(np.arange(G) == end_k, 0.0, -1e9)[None, :].repeat(n_paths, 0)  # [paths, G]
    policy = np.zeros((n, G), dtype=int)  # not used for value; kept for diagnostics
    for t in range(n - 1, -1, -1):
        S = paths[:, t]
        X = np.column_stack([S ** k for k in range(basis + 1)])
        # continuation estimate for every grid level
        cont = np.zeros((n_paths, G))
        for g in range(G):
            y = V_next[:, g]
            if np.all(y <= -1e8):
                cont[:, g] = -1e9
                continue
            ok = y > -1e8
            if ok.sum() < basis + 2:
                cont[:, g] = np.where(ok, y, -1e9)
                continue
            beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
            cont[:, g] = X @ beta
        V_now = np.full((n_paths, G), -1e9)
        for g in range(G):
            best = np.full(n_paths, -1e9)
            best_cash = np.zeros(n_paths)
            best_next = np.full(n_paths, g)
            for a in range(-k_w, k_i + 1):          # a > 0 inject a*step (effective eff*a*step into store), a < 0 withdraw
                if a > 0:
                    g2 = g + a
                    if g2 > deal.grid:
                        continue
                    cash = -(S * (1 + deal.fuel_loss) + deal.inj_cost) * a * step
                elif a < 0:
                    g2 = g + a
                    if g2 < 0:
                        continue
                    cash = (S - deal.wd_cost) * (-a) * step
                else:
                    g2, cash = g, np.zeros(n_paths)
                total = df[t] * cash + cont[:, g2]
                better = total > best
                best = np.where(better, total, best)
                best_cash = np.where(better, df[t] * cash, best_cash)
                best_next = np.where(better, g2, best_next)
            # realised value along the path: cash now + realised future value at the chosen next level
            V_now[:, g] = best_cash + V_next[np.arange(n_paths), best_next]
            V_now[best <= -1e8, g] = -1e9
        V_next = V_now
    k0 = int(round(deal.start_inventory / step))
    vals = V_next[:, k0]
    return {"value": float(vals.mean()), "se": float(vals.std(ddof=1) / np.sqrt(n_paths)), "per_path": vals}


# ---- lattice dynamic programme --------------------------------------------------------------------------------------
def lattice_dp(deal: StorageDeal, model: SpotModel, df: np.ndarray, n_x: int = 41, width: float = 3.5, sigma_scale: float = 1.0):
    """Backward induction on a Tauchen grid for the OU deviation x (no jumps) and the inventory grid.
    sigma_scale = 0 gives the deterministic (intrinsic) problem on the forward curve."""
    dates = deal.dates
    tau = np.asarray((dates - model.t0).days, dtype=float) / 365.25
    seas = model.seasonal(dates)
    sig = model.sigma * sigma_scale
    # stationary sd of x for the grid
    sd_stat = sig / np.sqrt(2 * model.kappa) if sig > 0 else 1e-6
    xg = np.linspace(-width * sd_stat, width * sd_stat, n_x) if sig > 0 else np.array([0.0])
    n = deal.months
    step, k_i, k_w = _actions(deal)
    G = deal.grid + 1
    end_k = int(round(deal.end_inventory / step))
    V = np.where(np.arange(G) == end_k, 0.0, -1e9)[None, :].repeat(len(xg), 0)  # [x, G]

    def transition(dt):
        if sig == 0 or dt <= 0:
            return np.eye(len(xg))
        e = np.exp(-model.kappa * dt)
        sd = sig * np.sqrt((1 - e ** 2) / (2 * model.kappa))
        mids = (xg[1:] + xg[:-1]) / 2
        P = np.zeros((len(xg), len(xg)))
        for i, x in enumerate(xg):
            mu = x * e
            cdf = norm.cdf((mids - mu) / sd)
            P[i, 0] = cdf[0]
            P[i, 1:-1] = np.diff(cdf)
            P[i, -1] = 1 - cdf[-1]
        return P

    # initial state: x0 placed on the grid by interpolation weights
    for t in range(n - 1, -1, -1):
        dt_next = tau[t + 1] - tau[t] if t + 1 < n else 0.0
        P = transition(dt_next) if t + 1 < n else None
        cont = P @ V if P is not None else V                       # expected continuation [x, G]
        S = np.exp(seas[t] + xg) if sig > 0 else np.exp(seas[t] + model.x0 * np.exp(-model.kappa * max(tau[t], 0.0)) + np.zeros(1))
        V_now = np.full((len(xg), G), -1e9)
        for g in range(G):
            best = np.full(len(xg), -1e9)
            for a in range(-k_w, k_i + 1):
                if a > 0:
                    g2 = g + a
                    if g2 > deal.grid:
                        continue
                    cash = -(S * (1 + deal.fuel_loss) + deal.inj_cost) * a * step
                elif a < 0:
                    g2 = g + a
                    if g2 < 0:
                        continue
                    cash = (S - deal.wd_cost) * (-a) * step
                else:
                    g2, cash = g, np.zeros(len(xg))
                total = df[t] * cash + cont[:, g2]
                best = np.maximum(best, total)
            V_now[:, g] = best
        V = V_now
    # value at x0 (the deviation on the valuation date, carried to the first month by the transition)
    k0 = int(round(deal.start_inventory / step))
    dt0 = max(tau[0], 0.0)
    if sig > 0:
        e = np.exp(-model.kappa * dt0)
        sd = sig * np.sqrt((1 - e ** 2) / (2 * model.kappa)) if dt0 > 0 else 1e-9
        mids = (xg[1:] + xg[:-1]) / 2
        cdf = norm.cdf((mids - model.x0 * e) / sd)
        w = np.concatenate([[cdf[0]], np.diff(cdf), [1 - cdf[-1]]])
        return {"value": float(w @ V[:, k0]), "grid_x": len(xg)}
    return {"value": float(V[0, k0]), "grid_x": 1}
