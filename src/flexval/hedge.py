"""Delta hedging and residual risk.

A sold European call on a monthly forward is delta-hedged with the forward on simulated daily paths, rebalanced at a
chosen frequency, with the delta taken from Black-76 on the forward implied by the path's state. The hedged P&L
distribution shows what is left after hedging: discrete rebalancing, jumps the Black-76 delta does not see, and the
gap between the model's implied volatility and realised volatility. VaR and Expected Shortfall of hedged and unhedged
P&L are reported in the deal's currency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import SpotModel
from .options import black76


def forward_from_state(model: SpotModel, x_t: np.ndarray, t_years: float, T_years: float, T_date, jumps: bool = True) -> np.ndarray:
    tau = max(T_years - t_years, 0.0)
    return model.forward_from_state(x_t, tau, model.seasonal([T_date])[0], jumps)


def delta_vec(F: np.ndarray, K: float, T: float, sigma: float, r: float) -> np.ndarray:
    from scipy.stats import norm
    v = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    return np.exp(-r * T) * norm.cdf(d1)


def var_es(pnl: np.ndarray, level: float = 0.99) -> tuple[float, float]:
    q = np.quantile(pnl, 1 - level)
    return float(-q), float(-pnl[pnl <= q].mean())


def hedge_call(model: SpotModel, expiry: str, K: float, volume: float, n_paths: int = 4000, rebalance_days: int = 5,
               seed: int = 7, jumps: bool = True, r: float = 0.04) -> dict:
    """Sell a call on the forward for `expiry` (a month start), volume units; hedge with the forward."""
    T_date = pd.Timestamp(expiry)
    days = pd.bdate_range(model.t0 + pd.Timedelta(days=1), T_date)
    T = (T_date - model.t0).days / 365.25
    S, X = model.simulate(days, n_paths, seed=seed, jumps=jumps, return_x=True)
    t_years = np.asarray((days - model.t0).days, dtype=float) / 365.25
    sigma_T = model.implied_vol([T_date])[0]
    F0 = model.forward([T_date], jumps=jumps)[0]
    prem = black76(model.forward([T_date], jumps=False)[0], K, T, sigma_T, r)   # Black-76 on the no-jump forward: the desk's quick price
    premium = prem.value * volume
    delta = prem.greeks["delta"]
    # forward price seen on each day from the path state
    F = np.column_stack([forward_from_state(model, X[:, i], t_years[i], T, T_date, jumps) for i in range(len(days))])
    hedge_pnl = np.zeros(n_paths)
    pos = np.full(n_paths, delta * volume)
    F_prev = np.full(n_paths, F0)
    n_rebal = 0
    for i in range(len(days)):
        hedge_pnl += pos * (F[:, i] - F_prev)
        F_prev = F[:, i]
        if (i + 1) % rebalance_days == 0 and i < len(days) - 1:
            tau = max(T - t_years[i], 1e-6)
            v = model.sigma ** 2 * (1 - np.exp(-2 * model.kappa * tau)) / (2 * model.kappa)   # remaining variance of ln S_T
            pos = delta_vec(F[:, i], K, tau, np.sqrt(v / tau), r) * volume
            n_rebal += 1
    payoff = np.maximum(S[:, -1] - K, 0.0) * volume
    df = np.exp(-r * T)
    unhedged = premium - df * payoff
    hedged = premium - df * payoff + hedge_pnl
    out = {"forward": float(F0), "strike": K, "expiry": str(T_date.date()), "volume": volume, "premium": float(premium),
           "delta0": float(delta), "implied_vol": float(sigma_T), "rebalances": n_rebal, "jumps": jumps}
    for name, p in (("unhedged", unhedged), ("hedged", hedged)):
        v95, e95 = var_es(p, 0.95)
        v99, e99 = var_es(p, 0.99)
        out[name] = {"mean": float(p.mean()), "sd": float(p.std(ddof=1)), "var95": v95, "es95": e95, "var99": v99, "es99": e99,
                     "worst5pct": float(np.quantile(p, 0.05))}
    out["sd_reduction"] = 1 - out["hedged"]["sd"] / out["unhedged"]["sd"]
    out["per_path_hedged"] = hedged
    out["per_path_unhedged"] = unhedged
    return out


def distribution(per_path: np.ndarray, label: str) -> dict:
    v95, e95 = var_es(per_path - per_path.mean(), 0.95)
    v99, e99 = var_es(per_path - per_path.mean(), 0.99)
    return {"label": label, "mean": float(per_path.mean()), "sd": float(per_path.std(ddof=1)),
            "p5": float(np.quantile(per_path, 0.05)), "p95": float(np.quantile(per_path, 0.95)),
            "var95_vs_mean": v95, "es95_vs_mean": e95, "var99_vs_mean": v99, "es99_vs_mean": e99}
