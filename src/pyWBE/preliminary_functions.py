"""
=====================
Preliminary Functions
=====================

Contains preliminary functions used to aid data analysis.

Note: Add type-hints and docstrings to functions as they are implemented.

"""


from pyWBE.exceptions import FunctionNotImplementedError
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt


def plot_time_series(series_x: pd.Series, series_y: pd.Series, plot_type: str = "linear") -> None:
    """
    This function plots the given time-series data for easy visualization.

    :param series_x: The independent variable, usually indicating time
    steps in arbitrary or specific units
    :type series_x: Pandas Series
    :param series_y: The dependent variable, indicating values of the
    variable of interest over time
    :type series_y: Pandas Series (of type float or int)
    :param plot_type: Determines the type of plot that is plotted.
    It can be either 'linear' (default) or 'log'. 'linear' plots
    series_y v/s series_x, 'log' plots the natural log of
    series_y v/s series_x
    :type plot_type: str
    """

    if plot_type == "linear":
        plt.plot(series_x, series_y)
        plt.xlabel(series_x.name)
        plt.ylabel(series_y.name)
        plt.title("Time Series Visualization")
        plt.show()
    elif plot_type == "log":
        plt.plot(series_x, np.log(series_y))
        plt.xlabel(series_x.name)
        plt.ylabel(f"{series_y.name} (log)")
        plt.title("Time Series Visualization (log scale)")
        plt.show()
    else:
        raise ValueError(f"The 'plot_type' parameter can only be 'linear' or 'log'. {plot_type} is invalid.")


def calculate_weekly_concentration_perc_change(conc_data: pd.Series) -> pd.Series:
    """
    This function computes the weekly percentage change in concentration levels
    in the given time-series data.
    :param conc_data: The concentration data, assumed to have a periodicity of 1 week
    :type conc_data: Pandas Series (of type float or int)
    :return: Returns the weekly percentage change in concentration levels
    :rtype: pd.Series
    """

    shifted_series = conc_data.copy(deep=True)
    shifted_series.iloc[:-1] = conc_data.iloc[1:]

    perc_change = (shifted_series - conc_data) / conc_data * 100

    return perc_change.iloc[:-1]


def analyze_trends(data: pd.Series) -> list[float]:
    """
    This function computes the trend line for the given data.\n
    :param data: The time-series data (assumed to be sorted
    in an increasing order of time).\n
    :type data: pd.Series\n
    :return: Returns the trend line values which can be plotted
    as date v/s returned trend line values.\n
    :rtype: list
    """
    z = np.polyfit(range(len(data)), data, 1)
    p = np.poly1d(z)
    trend_vals = p(range(len(data)))
    return list(trend_vals)


def change_point_detection(data: pd.Series, model: str = "l2",
                           min_size: int = 28, penalty: int = 1):
    """
    This function uses the PELT (Pruned Exact Linear Time) function
    of the Ruptures library to analyze the given time-series data
    for change point detection.\n
    :param data: A Pandas Series containing the time-series data
    whose change points need to be detected.\n
    :type data: pd.Series\n
    :param model: The model used by PELT to perform the analysis.
    Allowed types include "l1", "l2", and "rbf".\n
    :type model: str\n
    :param min_size: The minimum separation (time steps) between
    two consecutive change points detected by the model.\n
    :type min_size: int\n
    :param penalty: The penalty value used during prediction of
    change points.\n
    :type penalty: int\n
    :return: Returns a sorted list of breakpoints.\n
    :rtype: list\n
    """
    algo = rpt.Pelt(model=model, min_size=min_size).fit(data)
    result = algo.predict(pen=penalty)
    return result


def normalize_viral_load(data, normalization_type):
    raise FunctionNotImplementedError("""The function to normalize viral load has not been implemented.""")


def forecast_single_instance(data, model_type):
    raise FunctionNotImplementedError("""The function to forecast single instances from data has not been implemented.""")


def detect_seasonality(data):
    raise FunctionNotImplementedError("""The function to detect seasonality in data has not been implemented.""")
