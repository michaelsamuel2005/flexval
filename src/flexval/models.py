"""Spot price model: seasonal level plus a mean-reverting (Ornstein-Uhlenbeck) log deviation with jumps.

    ln S_t = f(t) + X_t,   f(t) = a + b t + sum_k [c_k cos(2 pi k t) + d_k sin(2 pi k t)],  k = 1, 2
    dX_t   = -kappa X_t dt + sigma dW_t + J dN_t,   J ~ N(mu_J, sigma_J^2),  N ~ Poisson(lambda)

This is the one-factor model of Lucia and Schwartz (2002) with Merton-style jumps. Calibration is on daily data
(dt = 1/252): ordinary least squares for f(t), an AR(1) regression for kappa and sigma, and jumps identified as
standardised AR(1) residuals beyond 3 robust standard deviations. The forward curve is model-implied,
F(t, T) = E[S_T | X_t], under the stated assumption that the pricing measure equals the physical one with no risk
premium; it must be replaced by exchange settlements when they are available. Without jumps, S_T is lognormal, so
Black-76 and Monte Carlo must agree, which is the first check in the scorecard.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

DT = 1.0 / 252.0


@dataclass
class SpotModel:
    name: str
    unit: str
    a: float
    b: float
    c: tuple          # cos coefficients, k = 1, 2
    d: tuple          # sin coefficients, k = 1, 2
    kappa: float      # per year
    sigma: float      # per sqrt(year)
    lam: float        # jumps per year
    mu_j: float
    sigma_j: float
    t0: pd.Timestamp  # calibration end (valuation date)
    x0: float         # deviation at t0
    s0: float
    n_obs: int
    r2_season: float
    phi: float        # daily AR(1) coefficient

    def to_dict(self) -> dict:
        d = asdict(self)
        d["t0"] = str(self.t0.date())
        return d

    # ---- deterministic part --------------------------------------------------------------------------------------
    def years(self, dates) -> np.ndarray:
        dates = pd.DatetimeIndex(dates)
        return np.asarray((dates - self.t0).days, dtype=float) / 365.25

    def seasonal(self, dates) -> np.ndarray:
        t = self.years(dates) + self._t0_years
        out = self.a + self.b * t
        for k in (1, 2):
            out = out + self.c[k - 1] * np.cos(2 * np.pi * k * t) + self.d[k - 1] * np.sin(2 * np.pi * k * t)
        return out

    @property
    def _t0_years(self) -> float:
        return float(self.t0.year + (self.t0.dayofyear - 1) / 365.25)

    # ---- forward curve and implied volatility -----------------------------------------------------------------------
    def forward(self, dates, jumps: bool = True) -> np.ndarray:
        """Model-implied forward price E[S_T | X_t0] for each date."""
        tau = np.maximum(self.years(dates), 0.0)
        return self.forward_from_state(self.x0, tau, self.seasonal(dates), jumps)

    def forward_from_state(self, x, tau, seasonal_T, jumps: bool = True) -> np.ndarray:
        """E[S_T | X_t = x] with tau = T - t years; x may be an array of states (then tau is a scalar)."""
        tau = np.asarray(tau, dtype=float)
        m = x * np.exp(-self.kappa * tau)
        v = self.sigma ** 2 * (1 - np.exp(-2 * self.kappa * tau)) / (2 * self.kappa)
        f = seasonal_T + m + 0.5 * v
        if jumps and self.lam > 0:
            f = f + self.jump_term(tau)
        return np.exp(f)

    def jump_term(self, tau) -> np.ndarray:
        """ln E[exp(sum of decayed jumps over tau)] to first order in lambda."""
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        s = np.linspace(0, 1, 65)
        extra = np.zeros_like(tau)
        for i, T in enumerate(tau):
            if T <= 0:
                continue
            u = s * T
            w = np.exp(-self.kappa * (T - u))
            integrand = np.exp(self.mu_j * w + 0.5 * self.sigma_j ** 2 * w ** 2) - 1.0
            extra[i] = self.lam * np.trapezoid(integrand, u)
        return extra if extra.size > 1 else extra[0]

    def _unused_forward_body(self, dates, jumps):
        tau = np.maximum(self.years(dates), 0.0)
        m = self.x0 * np.exp(-self.kappa * tau)
        v = self.sigma ** 2 * (1 - np.exp(-2 * self.kappa * tau)) / (2 * self.kappa)
        f = self.seasonal(dates) + m + 0.5 * v
        if jumps and self.lam > 0:
            # jumps decay at rate kappa; E[exp(J e^{-k s})] integrated over arrival times, to first order in lambda
            s = np.linspace(0, 1, 65)
            extra = np.zeros_like(tau)
            for i, T in enumerate(tau):
                if T <= 0:
                    continue
                u = s * T
                w = np.exp(-self.kappa * (T - u))
                integrand = np.exp(self.mu_j * w + 0.5 * self.sigma_j ** 2 * w ** 2) - 1.0
                extra[i] = self.lam * np.trapezoid(integrand, u)
            f = f + extra
        return np.exp(f)

    def implied_vol(self, dates) -> np.ndarray:
        """Black-76 volatility of ln S_T implied by the OU part: sqrt(Var[ln S_T] / tau)."""
        tau = np.maximum(self.years(dates), 1e-9)
        v = self.sigma ** 2 * (1 - np.exp(-2 * self.kappa * tau)) / (2 * self.kappa)
        return np.sqrt(v / tau)

    # ---- simulation ---------------------------------------------------------------------------------------------------
    def simulate(self, dates, n_paths: int, seed: int = 0, jumps: bool = True, antithetic: bool = True,
                 shocks: np.ndarray | None = None, return_x: bool = False):
        """Paths of S on the given dates (exact OU transitions). Returns array [n_paths, len(dates)].
        `shocks` lets a caller pass correlated standard normals [n_paths, len(dates)]."""
        dates = pd.DatetimeIndex(dates)
        tau = self.years(dates)
        rng = np.random.default_rng(seed)
        n = len(dates)
        if shocks is None:
            half = n_paths // 2 if antithetic else n_paths
            z = rng.standard_normal((half, n))
            z = np.vstack([z, -z]) if antithetic else z
            z = z[:n_paths]
        else:
            z = shocks
        x = np.empty((n_paths, n))
        prev_t, prev_x = 0.0, np.full(n_paths, self.x0)
        for i in range(n):
            dt = max(tau[i] - prev_t, 0.0)
            e = np.exp(-self.kappa * dt)
            sd = self.sigma * np.sqrt((1 - e ** 2) / (2 * self.kappa)) if dt > 0 else 0.0
            xi = prev_x * e + sd * z[:, i]
            if jumps and self.lam > 0 and dt > 0:
                nj = rng.poisson(self.lam * dt, n_paths)
                # each jump arrives uniformly in (prev_t, t] and decays until t
                tot = np.zeros(n_paths)
                k = nj.max()
                for j in range(k):
                    has = nj > j
                    u = rng.uniform(0, dt, n_paths)
                    jsize = rng.normal(self.mu_j, self.sigma_j, n_paths)
                    tot += np.where(has, jsize * np.exp(-self.kappa * (dt - u)), 0.0)
                xi = xi + tot
            x[:, i] = xi
            prev_t, prev_x = tau[i], xi
        S = np.exp(self.seasonal(dates)[None, :] + x)
        return (S, x) if return_x else S


def calibrate(series: pd.Series, name: str, unit: str, jump_z: float = 3.0) -> SpotModel:
    s = series.dropna()
    s = s[s > 0]
    y = np.log(s.values)
    t_abs = np.asarray(s.index.year + (s.index.dayofyear - 1) / 365.25, dtype=float)
    X = np.column_stack([np.ones_like(t_abs), t_abs] + [fn(2 * np.pi * k * t_abs) for k in (1, 2) for fn in (np.cos, np.sin)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    r2 = 1 - resid.var() / y.var()
    # AR(1) on the deviation
    x1, x0 = resid[1:], resid[:-1]
    phi = float((x1 * x0).sum() / (x0 * x0).sum())
    eps = x1 - phi * x0
    mad = np.median(np.abs(eps - np.median(eps))) * 1.4826
    jumps = np.abs(eps) > jump_z * mad
    n = len(eps)
    lam = float(jumps.sum() / (n * DT))
    mu_j = float(eps[jumps].mean()) if jumps.any() else 0.0
    sigma_j = float(eps[jumps].std(ddof=1)) if jumps.sum() > 1 else 0.0
    kappa = float(-np.log(max(phi, 1e-6)) / DT)
    sd_eps = float(eps[~jumps].std(ddof=1))
    sigma = float(sd_eps * np.sqrt(2 * kappa / (1 - phi ** 2)))
    return SpotModel(name=name, unit=unit, a=float(beta[0]), b=float(beta[1]), c=(float(beta[2]), float(beta[4])),
                     d=(float(beta[3]), float(beta[5])), kappa=kappa, sigma=sigma, lam=lam, mu_j=mu_j, sigma_j=sigma_j,
                     t0=s.index[-1], x0=float(resid[-1]), s0=float(s.values[-1]), n_obs=int(len(s)), r2_season=float(r2),
                     phi=phi)


def correlated_shocks(n_paths: int, n_steps: int, rho: float, seed: int, antithetic: bool = True):
    rng = np.random.default_rng(seed)
    half = n_paths // 2 if antithetic else n_paths
    z1 = rng.standard_normal((half, n_steps))
    z2 = rng.standard_normal((half, n_steps))
    if antithetic:
        z1, z2 = np.vstack([z1, -z1]), np.vstack([z2, -z2])
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2
    return z1[:n_paths], z2[:n_paths]


def terminal_correlation(m1: SpotModel, m2: SpotModel, rho: float, tau) -> np.ndarray:
    """Correlation of the OU deviations X1_T and X2_T at horizon tau when the Brownian shocks have correlation rho
    and the speeds of mean reversion differ."""
    tau = np.maximum(np.asarray(tau, dtype=float), 1e-9)
    k1, k2 = m1.kappa, m2.kappa
    num = 2 * np.sqrt(k1 * k2) / (k1 + k2) * (1 - np.exp(-(k1 + k2) * tau))
    den = np.sqrt((1 - np.exp(-2 * k1 * tau)) * (1 - np.exp(-2 * k2 * tau)))
    return rho * num / den


def residual_correlation(m1: SpotModel, s1: pd.Series, m2: SpotModel, s2: pd.Series) -> float:
    """Correlation of daily log-deviation changes of two series over their common dates."""
    j = pd.concat([s1.rename("a"), s2.rename("b")], axis=1).dropna()
    if len(j) < 100:
        return 0.0
    xa = np.log(j["a"].values) - m1.seasonal(j.index)
    xb = np.log(j["b"].values) - m2.seasonal(j.index)
    return float(np.corrcoef(np.diff(xa), np.diff(xb))[0, 1])
