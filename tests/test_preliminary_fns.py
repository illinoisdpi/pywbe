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
