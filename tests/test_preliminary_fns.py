import numpy as np
import pandas as pd
import pytest
from pyWBE import preliminary_functions
import ruptures as rpt


@pytest.mark.parametrize("test_input, expected", [
    (pd.Series(np.ones(10)), pd.Series(np.zeros(9))),
    (pd.Series([1, 2, 4, 8, 16, 32, 64]), pd.Series(np.ones(6)*100)),
    (pd.Series([2, 4, 7, 10.5, 13.125]), pd.Series([100.0, 75.0, 50.0, 25.0])),
    (pd.Series(np.arange(4)), pd.Series([np.inf, 100, 50])),
    (pd.Series(np.zeros(5)), pd.Series(np.ones(4) * np.nan))
])
def test_perc_conc_change(test_input, expected):
    assert preliminary_functions.calculate_weekly_concentration_perc_change(test_input).equals(expected)


def test_normalize_viral_load():
    data = pd.DataFrame({"to_normalize": [1, 2, 3, 4, 5], "normalize_by": [1, 2, 3, 4, 5]})
    assert preliminary_functions.normalize_viral_load(data, "to_normalize", "normalize_by").equals(pd.Series([1, 2, 3, 4, 5])/3)


@pytest.mark.parametrize("model", [
    ("l1"),
    ("l2"),
    ("rbf")
])
def test_change_point_detection(model):
    n_samples, dim, sigma = 1000, 3, 4
    n_bkps = 4  # number of breakpoints
    penalty = 10
    signal, bkps = rpt.pw_constant(n_samples, dim, n_bkps, noise_std=sigma)

    # detection
    algo = rpt.Pelt(model=model).fit(signal)
    result = algo.predict(pen=penalty)

    assert preliminary_functions.change_point_detection(signal, model, 2, penalty) == result
