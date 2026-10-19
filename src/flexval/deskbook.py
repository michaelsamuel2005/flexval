"""A small desk book with planted mispricings. Eight deals are booked with a value; four of them carry a classic
error (spot used instead of the forward, correlation left at zero, storage booked at intrinsic only, a heat rate in
the wrong units). The review re-prices every deal with flexval's methods and flags any that differ by more than the
tolerance. The scorecard requires all four planted errors found and no false flags on the four correct deals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import options, storage, tolling
from .models import SpotModel


def review_book(gas: SpotModel, power: SpotModel, gasgb: SpotModel, rho: float, r: float) -> dict:
    rows = []
    T_date = pd.Timestamp("2027-03-01")
    T = (T_date - gas.t0).days / 365.25
    F = gas.forward([T_date], jumps=False)[0]
    sig = gas.implied_vol([T_date])[0]
    K = round(F, 2)
    vol = 500_000

    def row(deal, booked, model, se, planted, why, tol_pct=0.02):
        tol = max(3 * se, tol_pct * abs(model))
        rows.append({"deal": deal, "booked_value": float(booked), "model_value": float(model), "tolerance": float(tol),
                     "difference": float(booked - model), "flagged": bool(abs(booked - model) > tol), "planted_error": planted, "note": why})

    # 1 correct European call
    c = options.black76(F, K, T, sig, r)
    row("Mar-27 HH call, 500k MMBtu", c.value * vol, c.value * vol, 0.0, False, "booked on the forward with the model volatility")
    # 2 planted: priced on spot instead of forward
    c_spot = options.black76(gas.s0, K, T, sig, r)
    row("Mar-27 HH call (second lot), 500k MMBtu", c_spot.value * vol, c.value * vol, 0.0, True, "booked using today's spot price in place of the Mar-27 forward")
    # 3 correct put by parity
    p = options.black76(F, K, T, sig, r, call=False)
    row("Mar-27 HH put, 500k MMBtu", p.value * vol, p.value * vol, 0.0, False, "booked by put-call parity from the call")
    # 4 correct Asian call (control variate MC)
    fix = pd.bdate_range("2027-03-01", "2027-03-31")
    paths = gas.simulate(fix, 20000, seed=31, jumps=False)
    Tn = (fix[-1] - gas.t0).days / 365.25
    a = options.asian_call_mc(paths, K, Tn, r, F=gas.forward([fix[-1]], jumps=False)[0], sigma=gas.implied_vol([fix[-1]])[0])
    a2 = options.asian_call_mc(gas.simulate(fix, 20000, seed=32, jumps=False), K, Tn, r, F=gas.forward([fix[-1]], jumps=False)[0], sigma=gas.implied_vol([fix[-1]])[0])
    row("Mar-27 HH Asian call, 500k MMBtu", a2.value * vol, a.value * vol, a.se * vol, False, "booked from an independent Monte Carlo run")
    # 5 planted: spread option with rho = 0
    Td = pd.Timestamp("2027-07-01")
    Tt = (Td - power.t0).days / 365.25
    Fp, Fg = power.forward([Td], jumps=False)[0], gasgb.forward([Td], jumps=False)[0]
    sp, sg = power.implied_vol([Td])[0], gasgb.implied_vol([Td])[0]
    hr, fixed, mwh = 2.0, 26.2, 400 * 24 * 31
    good = options.kirk(Fp, hr * Fg, fixed, Tt, sp, sg, rho, r)
    bad = options.kirk(Fp, hr * Fg, fixed, Tt, sp, sg, 0.0, r)
    row("Jul-27 spark-spread option strip, 400 MW", bad.value * mwh, good.value * mwh, 0.0, True, "booked with power-gas correlation 0 instead of 0.6")
    # 6 correct spread option
    row("Aug-27 spark-spread option strip, 400 MW", good.value * mwh, good.value * mwh, 0.0, False, "booked with the desk correlation")
    # 7 planted: storage booked at intrinsic only
    sd = storage.StorageDeal("book storage", "2026-11-01", 12, capacity=500_000, inj_rate=100_000, wd_rate=150_000, fuel_loss=0.01, grid=10)
    dfs = storage.discount_factors(sd, gas.t0)
    lp = storage.intrinsic_lp(sd, gas.forward(sd.dates, jumps=False), dfs)
    ls = storage.lsmc(sd, gas.simulate(sd.dates, 3000, seed=33, jumps=False), dfs)
    row("Storage 0.5 Bcf, Nov-26 to Oct-27", lp["value"], ls["value"], ls["se"], True, "booked at intrinsic value only; extrinsic value left out", tol_pct=0.05)
    # 8 planted: tolling heat rate in wrong units (7.0 MMBtu/MWh entered as the MWh/MWh heat rate)
    td = tolling.TollingDeal("book tolling", "2027-01-01", 365, mw=400, heat_rate=2.0, vom=4.0, emission_factor=0.37, carbon_price=60.0)
    cf = tolling.strip_closed_form(td, power, gasgb, rho)
    td_bad = tolling.TollingDeal("book tolling", "2027-01-01", 365, mw=400, heat_rate=7.0, vom=4.0, emission_factor=0.37, carbon_price=60.0)
    cf_bad = tolling.strip_closed_form(td_bad, power, gasgb, rho)
    row("CCGT tolling 2027, 400 MW", cf_bad["value"], cf["value"], 0.0, True, "heat rate entered as 7.0 (MMBtu per MWh) where the model takes MWh gas per MWh power (2.0)")
    return {"rows": rows, "n_planted": sum(r_["planted_error"] for r_ in rows), "n_flagged": sum(r_["flagged"] for r_ in rows)}
