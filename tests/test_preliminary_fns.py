import numpy as np
import pandas as pd
import pytest
from pyWBE import preliminary_functions


@pytest.mark.parametrize("test_input, expected", [
    (pd.Series(np.ones(10)), pd.Series(np.zeros(9))),
    (pd.Series([1, 2, 4, 8, 16, 32, 64]), pd.Series(np.ones(6)*100)),
    (pd.Series([2, 4, 7, 10.5, 13.125]), pd.Series([100.0, 75.0, 50.0, 25.0])),
    (pd.Series(np.arange(4)), pd.Series([np.inf, 100, 50])),
    (pd.Series(np.zeros(5)), pd.Series(np.ones(4) * np.nan))
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
    result = preliminary_functions.forecast_single_instance(test_input_data, test_window)
    assert result == pytest.approx(expected, abs=0.01)
