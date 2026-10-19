"""Deal economics the way an approval table shows them: expected value, cost of hedging, expected P&L after costs,
the worst 5% outcome, a capital charge and the return on it against a hurdle.

Capital is proxied by the 99% Expected Shortfall of the deal's P&L distribution relative to its mean (the loss a
risk committee would hold capital against). Hedging cost is a bid-offer charge on the hedge volume. Both are stated
assumptions, not EDF Trading's policy, and the memo shows the sensitivity to each.
"""
from __future__ import annotations

import numpy as np

from .hedge import var_es


def approval_table(name: str, currency: str, expected_value: float, per_path: np.ndarray, hedge_volume: float,
                   bid_offer: float, hurdle: float = 0.12, horizon_years: float = 1.0, fee_or_premium: float = 0.0) -> dict:
    """per_path: distribution of the deal's P&L (before fee/premium). fee_or_premium: what we receive (negative if we pay)."""
    hedge_cost = hedge_volume * bid_offer
    expected_pnl = expected_value + fee_or_premium - hedge_cost
    centred = per_path - per_path.mean()
    _, es99 = var_es(centred, 0.99)
    var95, es95 = var_es(centred, 0.95)
    capital = max(es99, 1e-9)
    roc = expected_pnl / capital / horizon_years
    worst5 = float(np.quantile(per_path, 0.05) + fee_or_premium - hedge_cost)
    return {"deal": name, "currency": currency, "expected_value": float(expected_value), "fee_or_premium": float(fee_or_premium),
            "hedge_volume": float(hedge_volume), "bid_offer": float(bid_offer), "hedge_cost": float(hedge_cost),
            "expected_pnl": float(expected_pnl), "worst_5pct": worst5, "var95": float(var95), "es95": float(es95),
            "capital_es99": float(capital), "return_on_capital": float(roc), "hurdle": hurdle,
            "meets_hurdle": bool(roc >= hurdle), "horizon_years": horizon_years}


def minimum_fee(expected_value: float, per_path: np.ndarray, hedge_volume: float, bid_offer: float, hurdle: float = 0.12,
                horizon_years: float = 1.0) -> float:
    """The fee (or premium) at which the deal just meets the hurdle."""
    centred = per_path - per_path.mean()
    _, es99 = var_es(centred, 0.99)
    return float(hurdle * es99 * horizon_years + hedge_volume * bid_offer - expected_value)
