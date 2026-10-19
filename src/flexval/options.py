"""Closed-form benchmarks and Monte Carlo pricers for the option-like payoffs in the book.

Black-76 (European options on a forward), Margrabe (exchange one asset for another: a spread option with zero strike),
Kirk's approximation (spread option with a strike), geometric-average Asian in closed form (used as a control variate
for the arithmetic Asian priced by Monte Carlo). Greeks by closed form where they exist and by central finite differences
otherwise. Every Monte Carlo result carries a standard error.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class Price:
    value: float
    se: float = 0.0
    method: str = ""
    greeks: dict | None = None

    def to_dict(self) -> dict:
        return {"value": self.value, "se": self.se, "method": self.method, "greeks": self.greeks or {}}


# ---- Black-76 -----------------------------------------------------------------------------------------------------------
def black76(F: float, K: float, T: float, sigma: float, r: float, call: bool = True) -> Price:
    if T <= 0 or sigma <= 0:
        intrinsic = max(F - K, 0.0) if call else max(K - F, 0.0)
        return Price(np.exp(-r * T) * intrinsic, 0.0, "Black-76 (intrinsic)")
    df = np.exp(-r * T)
    v = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    d2 = d1 - v
    if call:
        value = df * (F * norm.cdf(d1) - K * norm.cdf(d2))
        delta = df * norm.cdf(d1)
    else:
        value = df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        delta = -df * norm.cdf(-d1)
    gamma = df * norm.pdf(d1) / (F * v)
    vega = df * F * norm.pdf(d1) * np.sqrt(T)
    theta = -df * F * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * value
    return Price(float(value), 0.0, "Black-76", {"delta": float(delta), "gamma": float(gamma), "vega": float(vega),
                                                 "theta_per_year": float(theta)})


# ---- spread options ----------------------------------------------------------------------------------------------------
def margrabe(F1: float, F2: float, T: float, s1: float, s2: float, rho: float, r: float) -> Price:
    """Option to receive F1 and pay F2 (payoff max(S1 - S2, 0)); both lognormal."""
    sig = np.sqrt(s1 * s1 + s2 * s2 - 2 * rho * s1 * s2)
    if T <= 0 or sig <= 0:
        return Price(np.exp(-r * T) * max(F1 - F2, 0.0), 0.0, "Margrabe (intrinsic)")
    v = sig * np.sqrt(T)
    d1 = (np.log(F1 / F2) + 0.5 * v * v) / v
    d2 = d1 - v
    df = np.exp(-r * T)
    value = df * (F1 * norm.cdf(d1) - F2 * norm.cdf(d2))
    return Price(float(value), 0.0, "Margrabe", {"delta_1": float(df * norm.cdf(d1)), "delta_2": float(-df * norm.cdf(d2)),
                                                 "spread_vol": float(sig)})


def kirk(F1: float, F2: float, K: float, T: float, s1: float, s2: float, rho: float, r: float) -> Price:
    """Kirk (1995) approximation for max(S1 - S2 - K, 0)."""
    if K == 0:
        return margrabe(F1, F2, T, s1, s2, rho, r)
    a = F2 + K
    s2e = s2 * F2 / a
    sig = np.sqrt(s1 * s1 + s2e * s2e - 2 * rho * s1 * s2e)
    v = sig * np.sqrt(T)
    d1 = (np.log(F1 / a) + 0.5 * v * v) / v
    d2 = d1 - v
    df = np.exp(-r * T)
    value = df * (F1 * norm.cdf(d1) - a * norm.cdf(d2))
    return Price(float(value), 0.0, "Kirk", {"delta_1": float(df * norm.cdf(d1)), "delta_2": float(-df * norm.cdf(d2) * F2 / a),
                                             "spread_vol": float(sig)})


# ---- Asian ----------------------------------------------------------------------------------------------------------------
def geometric_asian_call(F: float, K: float, T: float, sigma: float, r: float, n: int) -> float:
    """Closed form for a call on the geometric average of n equally spaced fixings ending at T, under Black-76 dynamics.
    Used only as a control variate."""
    # average of ln S over fixings t_i = T i/n: mean ln F - 0.5 sigma^2 * mean(t_i), var = sigma^2 * sum_{i,j} min(t_i,t_j)/n^2
    t = T * np.arange(1, n + 1) / n
    mean = np.log(F) - 0.5 * sigma ** 2 * t.mean()
    var = sigma ** 2 * np.minimum.outer(t, t).sum() / n ** 2
    sd = np.sqrt(var)
    d1 = (mean - np.log(K) + var) / sd
    d2 = d1 - sd
    return float(np.exp(-r * T) * (np.exp(mean + 0.5 * var) * norm.cdf(d1) - K * norm.cdf(d2)))


def asian_call_mc(paths: np.ndarray, K: float, T: float, r: float, F: float | None = None, sigma: float | None = None,
                  control: bool = True) -> Price:
    """Arithmetic-average call priced from simulated fixings [n_paths, n_fixings]; geometric control variate when
    F and sigma are given (the lognormal benchmark model)."""
    df = np.exp(-r * T)
    arith = paths.mean(1)
    pay = df * np.maximum(arith - K, 0.0)
    n = paths.shape[1]
    if control and F is not None and sigma is not None:
        geo = np.exp(np.log(paths).mean(1))
        pay_g = df * np.maximum(geo - K, 0.0)
        cv = geometric_asian_call(F, K, T, sigma, r, n)
        beta = np.cov(pay, pay_g)[0, 1] / max(pay_g.var(), 1e-12)
        adj = pay - beta * (pay_g - cv)
        return Price(float(adj.mean()), float(adj.std(ddof=1) / np.sqrt(len(adj))), "Monte Carlo (geometric control variate)")
    return Price(float(pay.mean()), float(pay.std(ddof=1) / np.sqrt(len(pay))), "Monte Carlo")


def european_call_mc(ST: np.ndarray, K: float, T: float, r: float) -> Price:
    pay = np.exp(-r * T) * np.maximum(ST - K, 0.0)
    return Price(float(pay.mean()), float(pay.std(ddof=1) / np.sqrt(len(pay))), "Monte Carlo")


def spread_mc(S1: np.ndarray, S2: np.ndarray, K: float, T: float, r: float) -> Price:
    pay = np.exp(-r * T) * np.maximum(S1 - S2 - K, 0.0)
    return Price(float(pay.mean()), float(pay.std(ddof=1) / np.sqrt(len(pay))), "Monte Carlo")


def fd_greek(fn, x: float, h: float | None = None) -> float:
    h = h or max(1e-4 * abs(x), 1e-6)
    return (fn(x + h) - fn(x - h)) / (2 * h)
