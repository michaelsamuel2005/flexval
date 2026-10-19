# flexval

Pricing, hedging and risk of flexibility in energy markets: from a term sheet to a deal memo. Built by Michael Samuel with an AI coding assistant; every number is reproducible with `python -m flexval.run_all`.

## What it does
- One-factor seasonal mean-reverting log-price with jumps, calibrated to data (`models.py`).
- Options: Black-76, Margrabe, Kirk spread option, Asian option with control variate (`options.py`).
- Gas storage: LP intrinsic, rolling intrinsic, least-squares Monte Carlo, lattice DP (`storage.py`).
- Swing contract (`swing.py`), CCGT tolling agreement with start cost and minimum run (`tolling.py`), wind PPA with shape and volume risk (`ppa.py`).
- Delta hedging with rebalancing frequency and jumps, VaR/ES (`hedge.py`); deal approval table and minimum fee (`econ.py`).
- A desk book with four planted mispricings that the review must find, with no false flags (`deskbook.py`).
- `outputs/scorecard.md`: 21 checks, all passing. `excel/flexval_tool.xlsx`: live formulas that reconcile to Python to the cent. `docs/`: two deal memos.

## Data: what is real and what is not
| Series | Status |
|---|---|
| Henry Hub daily gas, 2008 to Oct 2026 | Real (US EIA, via datasets/natural-gas, ODC-PDDL) |
| GB day-ahead power | SIMULATED unless an Ember CSV is placed in `data/raw/`; then it is used automatically |
| NBP gas | PROXY: Henry Hub dynamics rescaled to a stated level (assumption) |
| Power-gas correlation 0.6, carbon price, hedging bid-offer, 12% hurdle | Assumptions |
| Wind capacity factors | SIMULATED (seasonal Beta) |

Tolling and PPA values therefore show the method, not a market view. Storage uses real Henry Hub prices but a hypothetical term sheet.

## Run
    pip install -e .[dev]
    python -m flexval.run_all        # about 25 s
    pytest -q tests
    python excel/build_workbook.py && python docs/build_memos.py

## Not done
No VBA macro in the workbook, no live market feeds, no credit or outage modelling.
