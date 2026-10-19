"""One command: calibrate, value every deal, hedge, compute deal economics, review the desk book, write outputs and
the scorecard.    python -m flexval.run_all [--quick]
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, models, options, storage, swing, tolling, ppa, hedge, econ
from .deskbook import review_book

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
TAB = OUT / "tables"


def jd(x):
    if isinstance(x, dict):
        return {k: jd(v) for k, v in x.items() if k not in ("per_path", "per_path_hedged", "per_path_unhedged")}
    if isinstance(x, (np.ndarray, list, tuple)):
        return [jd(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (pd.Timestamp,)):
        return str(x.date())
    return x


def main(quick: bool = False) -> dict:
    t_start = time.time()
    OUT.mkdir(exist_ok=True)
    TAB.mkdir(exist_ok=True)
    N = 2000 if quick else 6000
    checks: list[dict] = []

    def check(group, name, ok, detail):
        checks.append({"group": group, "check": name, "pass": bool(ok), "detail": detail})

    # ---- data and models ---------------------------------------------------------------------------------------------
    hh = data.henry_hub()
    gas = models.calibrate(hh, "Henry Hub", "US$/MMBtu")
    pw, pw_meta = data.gb_power()
    power = models.calibrate(pw, "GB power", "£/MWh")
    nbp, nbp_meta = data.nbp_proxy(hh, 30.0)
    gasgb = models.calibrate(nbp, "NBP proxy", "£/MWh")
    rho_est = models.residual_correlation(power, pw, gasgb, nbp)
    rho = 0.6  # stated assumption for the GB spread (see README); the estimate from the proxy is reported beside it
    res = {"valuation_date": str(gas.t0.date()), "data": {"henry_hub": data.henry_hub_provenance() | {
        "rows_used": gas.n_obs, "from": str(hh.index.min().date()), "to": str(hh.index.max().date())},
        "gb_power": pw_meta, "nbp_proxy": nbp_meta, "rho_power_gas_used": rho, "rho_power_gas_estimated_from_data": rho_est},
        "models": {"henry_hub": gas.to_dict(), "gb_power": power.to_dict(), "nbp_proxy": gasgb.to_dict()}}
    r = 0.04

    # forward curves
    months = pd.date_range("2026-11-01", periods=24, freq="MS")
    curve = pd.DataFrame({"month": months.strftime("%Y-%m"), "henry_hub_fwd_usd": gas.forward(months), "henry_hub_fwd_nojump_usd":
                          gas.forward(months, jumps=False), "henry_hub_implied_vol": gas.implied_vol(months),
                          "gb_power_fwd_gbp": power.forward(months), "gb_power_implied_vol": power.implied_vol(months),
                          "nbp_proxy_fwd_gbp": gasgb.forward(months)})
    curve.to_csv(TAB / "forward_curves.csv", index=False)
    res["forward_curve_first_12"] = curve.head(12).round(4).to_dict("records")

    # ---- 1. options on Henry Hub: benchmarks --------------------------------------------------------------------------
    T_date = pd.Timestamp("2027-02-01")
    T = (T_date - gas.t0).days / 365.25
    F0 = gas.forward([T_date], jumps=False)[0]
    sig = gas.implied_vol([T_date])[0]
    K = round(F0, 2)
    c76 = options.black76(F0, K, T, sig, r)
    p76 = options.black76(F0, K, T, sig, r, call=False)
    S_T = gas.simulate([T_date], 40000, seed=11, jumps=False)[:, 0]
    cmc = options.european_call_mc(S_T, K, T, r)
    check("Options", "Black-76 equals Monte Carlo for a European call (no jumps)", abs(cmc.value - c76.value) < 2.5 * cmc.se,
          f"Black-76 {c76.value:.4f}, MC {cmc.value:.4f} +/- {cmc.se:.4f}")
    parity = c76.value - p76.value - np.exp(-r * T) * (F0 - K)
    check("Options", "put-call parity", abs(parity) < 1e-10, f"C - P - df(F - K) = {parity:.2e}")
    S_Tj = gas.simulate([T_date], 40000, seed=11, jumps=True)[:, 0]
    cmcj = options.european_call_mc(S_Tj, K, T, r)
    # Asian on daily fixings in Feb 2027
    fix = pd.bdate_range("2027-02-01", "2027-02-26")
    Tn = (fix[-1] - gas.t0).days / 365.25
    paths = gas.simulate(fix, 20000, seed=12, jumps=False)
    Fn = gas.forward([fix[-1]], jumps=False)[0]
    sign = gas.implied_vol([fix[-1]])[0]
    a_plain = options.asian_call_mc(paths, K, Tn, r, control=False)
    a_cv = options.asian_call_mc(paths, K, Tn, r, F=Fn, sigma=sign, control=True)
    check("Options", "geometric control variate cuts the Asian standard error by half or more", a_cv.se <= 0.5 * a_plain.se,
          f"plain SE {a_plain.se:.5f}, control-variate SE {a_cv.se:.5f}")
    check("Options", "Asian call is worth less than the European call on the same forward and strike", a_cv.value < c76.value,
          f"Asian {a_cv.value:.4f} < European {c76.value:.4f}")
    res["options"] = {"forward_feb27": F0, "strike": K, "T_years": T, "implied_vol": sig, "european_call_black76": c76.to_dict(),
                      "european_put_black76": p76.to_dict(), "european_call_mc_nojumps": cmc.to_dict(),
                      "european_call_mc_with_jumps": cmcj.to_dict(), "jump_premium": cmcj.value - c76.value,
                      "asian_call_mc_plain": a_plain.to_dict(), "asian_call_mc_control_variate": a_cv.to_dict()}

    # ---- 2. storage ---------------------------------------------------------------------------------------------------
    sd = storage.StorageDeal("Henry Hub storage, Nov 2026 - Oct 2027", "2026-11-01", 12, capacity=1_000_000, inj_rate=200_000,
                             wd_rate=300_000, fuel_loss=0.01, grid=10)
    dfs = storage.discount_factors(sd, gas.t0)
    Fs = gas.forward(sd.dates, jumps=False)
    lp = storage.intrinsic_lp(sd, Fs, dfs)
    g0 = copy.copy(gas); g0.sigma = 1e-9; g0.lam = 0.0
    det = g0.simulate(sd.dates, 4, seed=1, jumps=False)
    lp_det = storage.intrinsic_lp(sd, det[0], dfs)
    ls_det = storage.lsmc(sd, det, dfs)
    lat_det = storage.lattice_dp(sd, gas, dfs, sigma_scale=0.0)
    check("Storage", "with zero volatility LP, LSMC and lattice agree to the dollar", abs(ls_det["value"] - lp_det["value"]) < 1
          and abs(lat_det["value"] - lp_det["value"]) < 1, f"LP {lp_det['value']:,.0f}, LSMC {ls_det['value']:,.0f}, lattice {lat_det['value']:,.0f}")
    S_m, X_m = gas.simulate(sd.dates, N, seed=2, jumps=False, return_x=True)
    ls = storage.lsmc(sd, S_m, dfs)
    lat = storage.lattice_dp(sd, gas, dfs)
    S_j = gas.simulate(sd.dates, N, seed=2, jumps=True)
    ls_j = storage.lsmc(sd, S_j, dfs)
    ri = storage.rolling_intrinsic(sd, gas, S_m[: (300 if quick else 800)], X_m[: (300 if quick else 800)], dfs)
    g15 = copy.copy(gas); g15.sigma = gas.sigma * 1.5
    ls15 = storage.lsmc(sd, g15.simulate(sd.dates, N, seed=2, jumps=False), dfs)
    check("Storage", "extrinsic value is positive: LSMC above intrinsic", ls["value"] > lp["value"],
          f"LSMC {ls['value']:,.0f} > intrinsic {lp['value']:,.0f}")
    check("Storage", "LSMC (a lower bound) does not exceed the lattice by more than 3 SE", ls["value"] <= lat["value"] + 3 * ls["se"],
          f"LSMC {ls['value']:,.0f} +/- {ls['se']:,.0f}, lattice {lat['value']:,.0f}")
    check("Storage", "value rises with volatility", ls15["value"] > ls["value"], f"sigma x1.5: {ls15['value']:,.0f} > {ls['value']:,.0f}")
    sched = pd.DataFrame({"month": sd.dates.strftime("%Y-%m"), "forward": Fs, "inject": lp["inj"], "withdraw": lp["wd"], "inventory": lp["inventory"]})
    sched.to_csv(TAB / "storage_intrinsic_schedule.csv", index=False)
    res["storage"] = {"deal": sd.to_dict(), "intrinsic_lp": lp["value"], "intrinsic_schedule": sched.round({"forward": 3, "inject": 0, "withdraw": 0, "inventory": 0}).to_dict("records"),
                      "lsmc": {"value": ls["value"], "se": ls["se"], "paths": N}, "lattice": lat, "lsmc_with_jumps": {"value": ls_j["value"], "se": ls_j["se"]},
                      "rolling_intrinsic_policy": {"value": ri["value"], "se": ri["se"], "paths": len(ri["per_path"])},
                      "extrinsic_lsmc": ls["value"] - lp["value"], "lsmc_sigma_x1_5": ls15["value"],
                      "deterministic_check": {"lp": lp_det["value"], "lsmc": ls_det["value"], "lattice": lat_det["value"]},
                      "hedged_distribution": hedge.distribution(ri["per_path"], "rolling intrinsic: intrinsic locked with forwards, re-hedged monthly"),
                      "unhedged_distribution": hedge.distribution(ls["per_path"], "spot dispatch (LSMC policy), no forward hedge")}

    # ---- 3. swing -------------------------------------------------------------------------------------------------------
    sw = swing.SwingDeal("Henry Hub swing, Nov 2026 - Oct 2027", "2026-11-01", 12, strike=2.85, q_min=50_000, q_max=150_000,
                         annual_min=900_000, annual_max=1_400_000)
    dfw = swing.discount(sw, gas.t0)
    sw_lp = swing.intrinsic_lp(sw, gas.forward(sw.dates, jumps=False), dfw)
    sw_lp_det = swing.intrinsic_lp(sw, det[0], dfw)
    sw_ls_det = swing.lsmc(sw, det, dfw)
    sw_ls = swing.lsmc(sw, S_m, dfw)
    check("Swing", "with zero volatility LP and LSMC agree to the dollar", abs(sw_ls_det["value"] - sw_lp_det["value"]) < 1,
          f"LP {sw_lp_det['value']:,.0f}, LSMC {sw_ls_det['value']:,.0f}")
    check("Swing", "flexibility has value: LSMC above intrinsic", sw_ls["value"] > sw_lp["value"], f"{sw_ls['value']:,.0f} > {sw_lp['value']:,.0f}")
    res["swing"] = {"deal": sw.to_dict(), "intrinsic_lp": sw_lp["value"], "intrinsic_take": list(sw_lp["take"]), "lsmc": {"value": sw_ls["value"], "se": sw_ls["se"]},
                    "flex_value": sw_ls["value"] - sw_lp["value"], "deterministic_check": {"lp": sw_lp_det["value"], "lsmc": sw_ls_det["value"]}}

    # ---- 4. tolling -----------------------------------------------------------------------------------------------------
    td = tolling.TollingDeal("GB CCGT tolling, calendar 2027", "2027-01-01", 365, mw=400, heat_rate=2.0, vom=4.0, emission_factor=0.37,
                             carbon_price=60.0, start_cost=15000.0, min_run_days=2)
    dft = tolling.discount(td, power.t0)
    cf = tolling.strip_closed_form(td, power, gasgb, rho)
    P, G = tolling.simulate_pair(td, power, gasgb, rho, N, 5, jumps=False)
    mc = tolling.strip_mc(td, P, G, dft)
    Pj, Gj = tolling.simulate_pair(td, power, gasgb, rho, N, 5, jumps=True)
    mcj = tolling.strip_mc(td, Pj, Gj, dft)
    dp = tolling.dispatch_dp(td, Pj, Gj, dft)
    td_free = copy.copy(td); td_free.start_cost = 0.0; td_free.min_run_days = 1
    dp_free = tolling.dispatch_dp(td_free, Pj, Gj, dft)
    check("Tolling", "Kirk strip (with the OU terminal correlation) equals the Monte Carlo strip within 1% or 3 SE",
          abs(cf["value"] - mc["value"]) < max(3 * mc["se"], 0.01 * mc["value"]),
          f"Kirk {cf['value']:,.0f}, MC {mc['value']:,.0f} +/- {mc['se']:,.0f}")
    check("Tolling", "constraints cost value: constrained dispatch below the unconstrained strip", dp["value"] <= mcj["value"] + 1e-6,
          f"constrained {dp['value']:,.0f} <= strip {mcj['value']:,.0f}")
    check("Tolling", "with no start cost and no minimum run the dispatch equals the strip exactly", abs(dp_free["value"] - mcj["value"]) < 1,
          f"{dp_free['value']:,.0f} vs {mcj['value']:,.0f}")
    rho_sens = {}
    for rr in (0.3, 0.6, 0.8):
        rho_sens[str(rr)] = tolling.strip_closed_form(td, power, gasgb, rr)["value"]
    hr_sens = {str(h): tolling.strip_closed_form(copy.copy(td).__class__(**{**td.to_dict(), "heat_rate": h}), power, gasgb, rho)["value"] for h in (1.8, 2.0, 2.2)}
    res["tolling"] = {"deal": td.to_dict(), "kirk_strip": cf["value"], "intrinsic_strip": cf["intrinsic"], "mc_strip_nojumps": {"value": mc["value"], "se": mc["se"]},
                      "mc_strip_with_jumps": {"value": mcj["value"], "se": mcj["se"], "run_days": mcj["run_days"]},
                      "constrained_dispatch": {"value": dp["value"], "se": dp["se"]}, "cost_of_constraints": mcj["value"] - dp["value"],
                      "unconstrained_dispatch_check": dp_free["value"], "rho_sensitivity": rho_sens, "heat_rate_sensitivity": hr_sens,
                      "delta_power_mwh_total": float(cf["delta_power_mwh"].sum()), "delta_gas_mwh_total": float(cf["delta_gas_mwh"].sum()),
                      "forward_power_avg": float(cf["forward_power"].mean()), "forward_gas_avg": float(cf["forward_gas"].mean())}
    pd.DataFrame({"date": td.dates, "forward_power": cf["forward_power"], "forward_gas": cf["forward_gas"], "kirk_value": cf["per_day"],
                  "delta_power_mwh": cf["delta_power_mwh"], "delta_gas_mwh": cf["delta_gas_mwh"]}).to_csv(TAB / "tolling_daily.csv", index=False)

    # ---- 5. PPA ----------------------------------------------------------------------------------------------------------
    pdeal = ppa.PPADeal("GB wind PPA, calendar 2027", "2027-01-01", 365, mw=100, strike=60.0)
    dfp = tolling.discount(td, power.t0)
    cfw = ppa.simulate_wind(pdeal, N, 21)
    pv = ppa.value(pdeal, Pj, cfw, dfp)
    pdeal0 = copy.copy(pdeal); pdeal0.cannibalisation = 0.0
    pv0 = ppa.value(pdeal0, Pj, cfw, dfp)
    check("PPA", "cannibalisation lowers the capture rate and the value", pv["capture_rate"] < pv0["capture_rate"] and pv["value"] < pv0["value"],
          f"capture rate {pv['capture_rate']:.3f} with, {pv0['capture_rate']:.3f} without (seasonal wind lifts it above 1)")
    pbe = copy.copy(pdeal); pbe.strike = pv["breakeven_strike"]
    pvb = ppa.value(pbe, Pj, cfw, dfp)
    check("PPA", "the deal is worth zero at the breakeven strike", abs(pvb["value"]) < 3 * pvb["se"] + 1, f"value at breakeven {pvb['value']:,.0f} +/- {pvb['se']:,.0f}")
    res["ppa"] = {"deal": pdeal.to_dict(), "value": pv["value"], "se": pv["se"], "capture_rate": pv["capture_rate"], "annual_volume_mwh": pv["annual_volume_mwh"],
                  "annual_volume_sd_mwh": pv["annual_volume_sd"], "breakeven_strike": pv["breakeven_strike"], "value_no_cannibalisation": pv0["value"],
                  "distribution": hedge.distribution(pv["per_path"], "fixed-price wind PPA, unhedged")}

    # ---- 6. hedging ------------------------------------------------------------------------------------------------------
    hedges = {}
    for jumps in (False, True):
        for rb in (1, 5, 21):
            h = hedge.hedge_call(gas, "2027-02-01", K=K, volume=1_000_000, n_paths=N, rebalance_days=rb, jumps=jumps, seed=7)
            hedges[f"jumps={jumps},rebalance={rb}d"] = {k: v for k, v in h.items() if not k.startswith("per_path")}
    h1, h21 = hedges["jumps=False,rebalance=1d"], hedges["jumps=False,rebalance=21d"]
    check("Hedging", "delta hedging cuts the P&L standard deviation", h1["hedged"]["sd"] < h1["unhedged"]["sd"],
          f"unhedged sd {h1['unhedged']['sd']:,.0f}, hedged daily {h1['hedged']['sd']:,.0f}")
    check("Hedging", "daily rebalancing beats monthly", h1["hedged"]["sd"] < h21["hedged"]["sd"], f"daily {h1['hedged']['sd']:,.0f} < monthly {h21['hedged']['sd']:,.0f}")
    hj = hedges["jumps=True,rebalance=1d"]
    jump_prem = (cmcj.value - c76.value) * 1_000_000
    check("Hedging", "with jumps, the mean hedged loss equals the jump premium left out of the Black-76 price (within 3 SE)",
          abs(hj["hedged"]["mean"] + jump_prem) < 3 * hj["hedged"]["sd"] / np.sqrt(N) + 3 * cmcj.se * 1_000_000,
          f"mean hedged P&L {hj['hedged']['mean']:,.0f}; jump premium {jump_prem:,.0f}")
    res["hedging"] = {"call_volume": 1_000_000, "premium_black76": c76.value * 1_000_000, "premium_mc_with_jumps": cmcj.value * 1_000_000, "jump_premium": jump_prem, "runs": hedges}
    pd.DataFrame([{"case": k, **{f"unhedged_{a}": v["unhedged"][a] for a in ("sd", "var99", "es99")}, **{f"hedged_{a}": v["hedged"][a] for a in ("mean", "sd", "var99", "es99")},
                   "sd_reduction": v["sd_reduction"]} for k, v in hedges.items()]).to_csv(TAB / "hedging.csv", index=False)

    # ---- 7. deal economics ------------------------------------------------------------------------------------------------
    tables = []
    st_fee = econ.minimum_fee(ri["value"], ri["per_path"], hedge_volume=float(lp["inj"].sum() + lp["wd"].sum()), bid_offer=0.01)
    tables.append(econ.approval_table("Storage (rolling-intrinsic hedge)", "USD", ri["value"], ri["per_path"], float(lp["inj"].sum() + lp["wd"].sum()),
                                      0.01, fee_or_premium=-st_fee * 0.8))
    tl_fee = econ.minimum_fee(dp["value"], dp["per_path"], hedge_volume=float(abs(cf["delta_power_mwh"]).sum() + abs(cf["delta_gas_mwh"]).sum()), bid_offer=0.25)
    tables.append(econ.approval_table("Tolling (constrained dispatch)", "GBP", dp["value"], dp["per_path"],
                                      float(abs(cf["delta_power_mwh"]).sum() + abs(cf["delta_gas_mwh"]).sum()), 0.25, fee_or_premium=-tl_fee * 1.1))
    tables.append(econ.approval_table("Wind PPA at 60 £/MWh", "GBP", pv["value"], pv["per_path"], pv["annual_volume_mwh"], 0.25))
    chk = econ.approval_table("fee check", "USD", ri["value"], ri["per_path"], float(lp["inj"].sum() + lp["wd"].sum()), 0.01, fee_or_premium=st_fee)
    check("Economics", "the minimum fee makes the return on capital equal to the hurdle", abs(chk["return_on_capital"] - chk["hurdle"]) < 1e-9,
          f"RoC {chk['return_on_capital']:.4f} at fee {st_fee:,.0f}")
    res["economics"] = {"tables": tables, "storage_min_fee": st_fee, "tolling_min_fee": tl_fee}
    pd.DataFrame(tables).to_csv(TAB / "approval_tables.csv", index=False)

    # ---- 8. desk book review --------------------------------------------------------------------------------------------
    book = review_book(gas, power, gasgb, rho, r)
    found = [b for b in book["rows"] if b["planted_error"] and b["flagged"]]
    false_flags = [b for b in book["rows"] if not b["planted_error"] and b["flagged"]]
    check("Desk book", "every planted mispricing in the desk book is flagged", len(found) == book["n_planted"], f"{len(found)} of {book['n_planted']} flagged")
    check("Desk book", "no correctly booked deal is flagged", len(false_flags) == 0, f"false flags: {[b['deal'] for b in false_flags]}")
    res["desk_book"] = book
    pd.DataFrame(book["rows"]).to_csv(TAB / "desk_book_review.csv", index=False)

    # ---- scorecard --------------------------------------------------------------------------------------------------------
    n_pass = sum(c["pass"] for c in checks)
    res["scorecard"] = {"n_checks": len(checks), "n_passed": n_pass, "checks": checks, "quick": quick, "elapsed_seconds": round(time.time() - t_start, 1)}
    (OUT / "results.json").write_text(json.dumps(jd(res), indent=1))
    lines = [f"# flexval scorecard: {n_pass} of {len(checks)} checks pass", "", "| group | check | pass | detail |", "|---|---|---|---|"]
    lines += [f"| {c['group']} | {c['check']} | {'yes' if c['pass'] else 'NO'} | {c['detail']} |" for c in checks]
    (OUT / "scorecard.md").write_text("\n".join(lines) + "\n")
    print(f"flexval: {n_pass}/{len(checks)} checks pass in {time.time() - t_start:.0f}s")
    return res


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
