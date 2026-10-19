import numpy as np, pytest
from flexval import options, run_all

def test_black76_put_call_parity():
    c = options.black76(3.0, 2.8, 0.5, 0.35, 0.04); p = options.black76(3.0, 2.8, 0.5, 0.35, 0.04, call=False)
    assert abs(c.value - p.value - np.exp(-0.04 * 0.5) * (3.0 - 2.8)) < 1e-12

def test_kirk_with_zero_strike_matches_margrabe():
    k = options.kirk(50, 40, 0.0, 1.0, 0.4, 0.3, 0.5, 0.0); m = options.margrabe(50, 40, 1.0, 0.4, 0.3, 0.5, 0.0)
    assert abs(k.value - m.value) < 1e-6

def test_black76_zero_vol_is_discounted_intrinsic():
    c = options.black76(3.0, 2.5, 1.0, 1e-9, 0.05)
    assert abs(c.value - np.exp(-0.05) * 0.5) < 1e-6

def test_full_scorecard_quick():
    res = run_all.main(quick=True)
    failed = [c for c in res["scorecard"]["checks"] if not c["pass"]]
    assert not failed, failed
