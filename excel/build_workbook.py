import json
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from flexval import options
r = json.load(open("outputs/results.json"))
h = r["hedging"]["runs"]["jumps=False,rebalance=1d"]
wb = Workbook(); bold = Font(bold=True); inp = PatternFill("solid", fgColor="FFF2CC")
def put(ws, rows):
    for i, row in enumerate(rows, 1):
        for j, v in enumerate(row, 1):
            c = ws.cell(i, j, v)
            if i == 1: c.font = bold
ws = wb.active; ws.title = "Black76"
put(ws, [["Black-76 call on a Henry Hub forward (live formulas)", "", ""],
 ["Forward F ($/MMBtu)", h["forward"], "input"], ["Strike K", h["strike"], "input"], ["Time to expiry T (years)", 0.3, "input"],
 ["Volatility", h["implied_vol"], "input (model-implied)"], ["Rate r", 0.04, "input"], ["Volume (MMBtu)", h["volume"], "input"],
 ["d1", "=(LN(B2/B3)+0.5*B5^2*B4)/(B5*SQRT(B4))", ""], ["d2", "=B8-B5*SQRT(B4)", ""],
 ["Call value $/MMBtu", "=EXP(-B6*B4)*(B2*NORMSDIST(B8)-B3*NORMSDIST(B9))", ""],
 ["Premium ($)", "=B10*B7", ""], ["Delta (per forward)", "=EXP(-B6*B4)*NORMSDIST(B8)", ""],
 ["Python premium ($)", None, "from flexval run_all"], ["Difference ($)", "=B11-B13", "should be ~0 once T is set to the exact value"]])
from datetime import date
T = (date(2027,2,1) - date(2026,10,14)).days / 365.25
ws["B4"] = T
py = options.black76(h["forward"], h["strike"], T, h["implied_vol"], 0.04).value * h["volume"]
ws["B13"] = py
for c in ("B2","B3","B4","B5","B6","B7"): ws[c].fill = inp
ws.column_dimensions["A"].width = 34; ws.column_dimensions["C"].width = 44

k = wb.create_sheet("Kirk")
t = r["tolling"]
Fp, Fg, hr, fx = t["forward_power_avg"], t["forward_gas_avg"], t["deal"]["heat_rate"], t["deal"]["vom"] + t["deal"]["emission_factor"] * t["deal"]["carbon_price"]
sp, sg, rho, T2 = 0.60, 0.50, 0.6, 0.5
put(k, [["Kirk spark-spread option (live formulas; illustrative inputs)", "", ""],
 ["Power forward £/MWh", Fp, "input"], ["Gas forward p/therm-equivalent £/MWh", Fg, "input"], ["Heat rate", hr, "input"], ["Fixed costs £/MWh", fx, "input"],
 ["Power vol", sp, "input"], ["Gas vol", sg, "input"], ["Correlation", rho, "input"], ["T (years)", T2, "input"], ["r", 0.04, "input"],
 ["F2 = HR x gas", "=B4*B3", ""], ["Strike-adjusted F2+K", "=B11+B5", ""], ["Effective vol", "=SQRT(B6^2-2*B8*B6*B7*B11/B12+(B7*B11/B12)^2)", ""],
 ["d1", "=(LN(B2/B12)+0.5*B13^2*B9)/(B13*SQRT(B9))", ""], ["d2", "=B14-B13*SQRT(B9)", ""],
 ["Option value £/MWh", "=EXP(-B10*B9)*(B2*NORMSDIST(B14)-B12*NORMSDIST(B15))", ""], ["Python value £/MWh", options.kirk(Fp, hr*Fg, fx, T2, sp, sg, rho, 0.04).value, "flexval.options.kirk"],
 ["Difference", "=B16-B17", "should be ~0"]])
for c in ("B2","B3","B4","B5","B6","B7","B8","B9","B10"): k[c].fill = inp
k.column_dimensions["A"].width = 38

a = wb.create_sheet("Approval")
e = r["economics"]["tables"][0]
put(a, [["Deal approval table (live formulas, storage example, USD)", ""],
 ["Expected value", e["expected_value"]], ["Fee received", e["fee_or_premium"]], ["Hedge volume", e["hedge_volume"]], ["Bid-offer per unit", e["bid_offer"]],
 ["Capital proxy: ES99 of P&L", e["capital_es99"]], ["Hurdle return", 0.12], ["Hedge cost", "=B4*B5"], ["Expected P&L", "=B2+B3-B8"],
 ["Return on capital", "=B9/B6"], ["Meets hurdle?", '=IF(B10>=B7,"yes","no")'], ["Minimum fee for hurdle", "=B7*B6+B8-B2"]])
for c in ("B2","B3","B4","B5","B6","B7"): a[c].fill = inp
a.column_dimensions["A"].width = 34

s = wb.create_sheet("Storage_intrinsic")
s.append(["month", "forward", "inject", "withdraw", "inventory"])
for x in r["storage"]["intrinsic_schedule"]: s.append([x["month"], x["forward"], x["inject"], x["withdraw"], x["inventory"]])
s.append([]); s.append(["LP intrinsic value $", r["storage"]["intrinsic_lp"]]); s.append(["LSMC value $", r["storage"]["lsmc"]["value"]]); s.append(["Extrinsic $", "=B%d-B%d" % (s.max_row, s.max_row-1)])
wb.save("excel/flexval_tool.xlsx")
