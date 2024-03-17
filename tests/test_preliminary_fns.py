import numpy as np
import pandas as pd
import pytest
from pyWBE import preliminary_functions
import math
import ruptures as rpt


@pytest.mark.parametrize("test_input, expected", [
    (pd.Series(np.ones(10), index=pd.date_range(start='01/01/2024', periods=10, freq='7D')),
     pd.Series(np.zeros(9), index=pd.date_range(start='01/01/2024', periods=9, freq='7D'))),
    (pd.Series([1, 2, 4, 8, 16, 32, 64], index=pd.date_range(start='01/01/2024', periods=7, freq='7D')),
     pd.Series(np.ones(6)*100, index=pd.date_range(start='01/01/2024', periods=6, freq='7D'))),
    (pd.Series([2, 4, 7, 10.5, 13.125], index=pd.date_range(start='01/01/2024', periods=5, freq='7D')),
     pd.Series([100.0, 75.0, 50.0, 25.0], index=pd.date_range(start='01/01/2024', periods=4, freq='7D'))),
    (pd.Series(np.arange(4), index=pd.date_range(start='01/01/2024', periods=4, freq='7D')),
     pd.Series([np.inf, 100, 50], index=pd.date_range(start='01/01/2024', periods=3, freq='7D'))),
    (pd.Series(np.zeros(5), index=pd.date_range(start='01/01/2024', periods=5, freq='7D')),
     pd.Series(np.ones(4) * np.nan, index=pd.date_range(start='01/01/2024', periods=4, freq='7D')))
])
def test_perc_conc_change(test_input, expected):
    assert preliminary_functions.calculate_weekly_concentration_perc_change(test_input).equals(expected)


@pytest.mark.parametrize("seed, num_vals, A, b, start_date, window_start_date, freq", [
    (0, 100, 0, 0, pd.Timestamp("02/06/2024"), pd.Timestamp("04/06/2024"), 'D'),
    (1, 60, 1, 1, pd.Timestamp("02/06/2024"), pd.Timestamp("03/06/2024"), 'D'),
    (1, 50, 2, 0, pd.Timestamp("02/06/2024"), pd.Timestamp("03/06/2024"), 'D')
])
def test_single_instance_forecast(seed, num_vals, A, b, start_date, window_start_date, freq):
    dates = pd.date_range(start=start_date, periods=num_vals, freq=freq)
    data = [seed]
    for _ in range(num_vals):
        data.append(A*data[-1] + b)
    series = pd.Series(data[:-1], index=dates)
    window = pd.date_range(start=window_start_date, end=dates.max(), freq=freq)
    test_input_data, test_window, expected = series, window, data[-1]
    result = preliminary_functions.forecast_single_instance(test_input_data, test_window).iloc[-1]
    assert result == pytest.approx(expected, abs=0.01)


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


def test_normalize_viral_load():
    data = pd.DataFrame({"to_normalize": [1, 2, 3, 4, 5], "normalize_by": [1, 2, 3, 4, 5]})
    assert preliminary_functions.normalize_viral_load(data, "to_normalize", "normalize_by").equals(pd.Series([1, 2, 3, 4, 5])/3)


@pytest.mark.parametrize("model", [
    ("l1"),
    ("l2"),
    ("rbf")
])
def test_change_point_detection(model):
    n_samples, dim, sigma = 1000, 1, 4
    n_bkps = 4  # number of breakpoints
    penalty = 10
    signal, bkps = rpt.pw_constant(n_samples, dim, n_bkps, noise_std=sigma)

    # detection
    algo = rpt.Pelt(model=model).fit(signal)
    result = algo.predict(pen=penalty)
    signal = pd.Series(signal.flatten(), index=pd.date_range(start="02/06/2024", periods=len(signal.flatten()), freq='D'))

    assert preliminary_functions.change_point_detection(signal, model, 2, penalty) == result
