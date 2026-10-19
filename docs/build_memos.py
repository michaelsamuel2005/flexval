import json
from docx import Document
from docx.shared import Pt, Cm
r = json.load(open("outputs/results.json"))
def doc(title):
    d = Document(); s = d.sections[0]; s.page_width, s.page_height = Cm(21), Cm(29.7)
    for m in ("left_margin","right_margin","top_margin","bottom_margin"): setattr(s, m, Cm(1.8))
    d.styles["Normal"].font.name = "Arial"; d.styles["Normal"].font.size = Pt(10)
    d.add_heading(title, 1); return d
def table(d, rows):
    t = d.add_table(rows=0, cols=len(rows[0])); t.style = "Light Grid Accent 1"
    for row in rows:
        c = t.add_row().cells
        for i, v in enumerate(row): c[i].text = str(v)
def m(x): return f"{x:,.0f}"

st = r["storage"]; e = r["economics"]["tables"][0]; sd = st["deal"]
d = doc("Deal memo: Henry Hub gas storage, Nov 2026 to Oct 2027")
d.add_paragraph("Status: DRAFT for discussion. Valuation date " + r["valuation_date"] + ". Gas prices are real EIA Henry Hub; the structure is a hypothetical term sheet, not a real counterparty.")
d.add_heading("Recommendation", 2)
d.add_paragraph(f"Value the deal at USD {m(st['lsmc']['value'])} (least-squares Monte Carlo, standard error {m(st['lsmc']['se'])}), of which USD {m(st['intrinsic_lp'])} is intrinsic and USD {m(st['extrinsic_lsmc'])} is extrinsic. Lock the intrinsic with forwards, re-optimise monthly (rolling intrinsic, mean USD {m(st['rolling_intrinsic_policy']['value'])}) and quote the extrinsic only at a discount: it exists only if the desk actually re-hedges. The minimum fee for a 12% return on capital is USD {m(r['economics']['storage_min_fee'])}; a negative figure means the deal clears the hurdle with no fee under these assumptions.")
d.add_heading("Term sheet", 2)
table(d, [["Item", "Value"], ["Capacity", m(sd["capacity"]) + " MMBtu"], ["Injection / withdrawal rate", f"{m(sd['inj_rate'])} / {m(sd['wd_rate'])} MMBtu per month"], ["Injection / withdrawal cost", f"{sd['inj_cost']} / {sd['wd_cost']} USD per MMBtu"], ["Fuel loss (in kind)", f"{sd['fuel_loss']:.0%}"], ["Start / end inventory", "0 / 0"], ["Discount rate", f"{sd['r']:.0%}"]])
d.add_heading("Valuation and cross-checks", 2)
table(d, [["Method", "Value USD", "Note"], ["Intrinsic (linear programme, forward curve)", m(st["intrinsic_lp"]), "exact for the given curve"], ["LSMC, no jumps", m(st["lsmc"]["value"]), f"SE {m(st['lsmc']['se'])}; lower bound"], ["Lattice dynamic programme", m(st["lattice"]["value"]), "within noise of LSMC"], ["LSMC with price jumps", m(st["lsmc_with_jumps"]["value"]), "jump risk raises option value"], ["LSMC, volatility x1.5", m(st["lsmc_sigma_x1_5"]), "value rises with volatility"]])
d.add_paragraph(f"Method check: with volatility set near zero, the linear programme, LSMC and lattice all give USD {m(st['deterministic_check']['lp'])}.")
d.add_heading("Risk and economics", 2)
table(d, [["Measure", "USD"], ["Expected P&L after hedge cost", m(e["expected_pnl"])], ["Worst 5% outcome", m(e["worst_5pct"])], ["VaR 95% vs mean", m(e["var95"])], ["ES 99% (capital proxy)", m(e["capital_es99"])], ["Return on capital", f"{e['return_on_capital']:.0%} against a {e['hurdle']:.0%} hurdle"]])
d.add_paragraph(f"Hedging with rolling intrinsic cut the 99% shortfall from USD {m(st['unhedged_distribution']['es99_vs_mean'])} to USD {m(st['hedged_distribution']['es99_vs_mean'])}: a modest reduction, because the extrinsic value is itself the exposure to price moves.")
d.add_heading("Limits and assumptions", 2)
for t in ["One-factor model with jumps calibrated to Henry Hub history since 2008; storage is priced on a monthly grid.", "Capital is an Expected Shortfall proxy and the 12% hurdle and USD 0.01 bid-offer are stated assumptions, not any firm's policy.", "Counterparty credit, operational constraints and seasonal injection limits beyond the monthly rates are not modelled.", "Henry Hub is a US price; it stands in for the structure and is not a European gas hub."]:
    d.add_paragraph(t, style="List Bullet")
d.save("docs/deal_memo_storage.docx")

t = r["tolling"]; e2 = r["economics"]["tables"][1]; td = t["deal"]
d = doc("Deal memo: GB CCGT tolling agreement, calendar 2027")
d.add_paragraph("Status: DRAFT, ILLUSTRATIVE. GB power prices in this build are SIMULATED (no Ember file supplied) and gas is a Henry Hub-based proxy, so levels below demonstrate the method and are not a market view.")
d.add_heading("Recommendation", 2)
d.add_paragraph(f"On these inputs the toll is worth GBP {m(t['constrained_dispatch']['value'])} with start costs and a two-day minimum run, GBP {m(t['mc_strip_with_jumps']['value'])} unconstrained, and GBP {m(t['intrinsic_strip'])} intrinsic. The physical constraints cost GBP {m(t['cost_of_constraints'])}. The large gap between intrinsic and total value is the optionality of running only when the spark spread is positive, about {t['mc_strip_with_jumps']['run_days']:.0f} days a year here. The size of the value reflects the simulated spread (power forward average GBP {t['forward_power_avg']:.1f}/MWh against gas GBP {t['forward_gas_avg']:.1f}/MWh), so the figure should not be quoted until real GB prices replace the simulation.")
d.add_heading("Term sheet", 2)
table(d, [["Item", "Value"], ["Capacity", f"{td['mw']} MW, {td['days']} days"], ["Heat rate", f"{td['heat_rate']} MWh gas per MWh power"], ["Variable O&M", f"GBP {td['vom']}/MWh"], ["Carbon", f"{td['emission_factor']} tCO2/MWh at GBP {td['carbon_price']}/t (fixed input)"], ["Start cost / minimum run", f"GBP {m(td['start_cost'])} / {td['min_run_days']} days"]])
d.add_heading("Valuation and cross-checks", 2)
table(d, [["Method", "GBP"], ["Kirk spread-option strip", m(t["kirk_strip"])], ["Monte Carlo strip, no jumps", f"{m(t['mc_strip_nojumps']['value'])} (SE {m(t['mc_strip_nojumps']['se'])})"], ["Monte Carlo strip, with jumps", m(t["mc_strip_with_jumps"]["value"])], ["Constrained dispatch", m(t["constrained_dispatch"]["value"])]])
d.add_paragraph("Kirk agrees with Monte Carlo within 1% once the correlation is converted to its terminal value under mean reversion. With no start cost and no minimum run the dispatch value equals the strip exactly.")
d.add_heading("Sensitivities", 2)
table(d, [["Input", "Value GBP"]] + [[f"Correlation {k}", m(v)] for k, v in t["rho_sensitivity"].items()] + [[f"Heat rate {k}", m(v)] for k, v in t["heat_rate_sensitivity"].items()])
d.add_paragraph("Higher correlation lowers the value, because the spread becomes less volatile; the correlation of 0.6 is an assumption, since the estimated power-gas residual correlation from the simulated series is not informative.")
d.add_heading("Hedge and limits", 2)
for s in [f"Hedge the delta with power and gas forwards: about {m(t['delta_power_mwh_total'])} MWh of power sold and {m(t['delta_gas_mwh_total'])} MWh of gas bought, rebalanced as spreads move.", "Price jumps are not hedgeable with forwards; the unhedged jump risk is shown in the hedging study (hedge.py).", "Carbon is a fixed price input; a carbon price series would add a third factor.", "No credit, ramp-rate or outage modelling."]:
    d.add_paragraph(s, style="List Bullet")
d.save("docs/deal_memo_tolling.docx")
