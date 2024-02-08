import numpy as np
import pandas as pd
import pytest
from pyWBE import preliminary_functions
import math


@pytest.mark.parametrize("test_input, expected", [
    (pd.Series(np.ones(10)), pd.Series(np.zeros(9))),
    (pd.Series([1, 2, 4, 8, 16, 32, 64]), pd.Series(np.ones(6)*100)),
    (pd.Series([2, 4, 7, 10.5, 13.125]), pd.Series([100.0, 75.0, 50.0, 25.0])),
    (pd.Series(np.arange(4)), pd.Series([np.inf, 100, 50])),
    (pd.Series(np.zeros(5)), pd.Series(np.ones(4) * np.nan))
])
def test_perc_conc_change(test_input, expected):
    assert preliminary_functions.calculate_weekly_concentration_perc_change(test_input).equals(expected)


@pytest.mark.parametrize("model_type", [
    ("additive"),
    ("multiplicative")
])
def test_seasonality_detection(model_type):
    s = np.cos(np.linspace(start=0, stop=10*2*math.pi, num=100))
    t = np.linspace(start=0, stop=2*math.pi, num=100)
    r = np.clip(np.random.normal(size=t.shape), 0, 0.3)

    if model_type == "additive":
        vals = s + t + r
    elif model_type == "multiplicative":
        vals = s * t * r + 2   # Adding 2 for scaling
    else:
        return
    dates = pd.date_range(start="02/06/2024", periods=len(vals), freq='D')
    series = pd.Series(vals, index=dates)

    output = preliminary_functions.detect_seasonality(series, model_type)

    diff_s = (output.seasonal - s).sum() / len(s)
    diff_t = (output.trend - t).sum() / len(t)
    diff_r = (output.resid - r).sum() / len(r)

    assert (diff_s + diff_t + diff_r) / 3 == pytest.approx(0, abs=0.35)
