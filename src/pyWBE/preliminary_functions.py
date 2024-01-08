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


def plot_time_series(series_x: pd.Series, series_y: pd.Series, plot_type: str = "linear") -> None:
    """
    This function plots the given time-series data for easy visualization.

    :param series_x: The independent variable, usually indicating time
    steps in arbitrary or specific units
    :type series_x: pd.Series
    :param series_y: The dependent variable, indicating values of the
    variable of interest over time
    :type series_y: pd.Series
    :param plot_type: Determines the type of plot that is plotted.
    It can be either 'linear' (default) or 'log'
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


def calculate_weekly_concentration_perc_change(conc_data: pd.Series):
    raise FunctionNotImplementedError("""The function to calculate weekly percentage
                                      changes in concentration has not been implemented.""")


def analyze_trends(data, analysis_type):
    raise FunctionNotImplementedError("""The function to analyze trends has not been implemented.""")


def change_point_detection(time_instance, method):
    raise FunctionNotImplementedError("""The function to change point detection has not been implemented.""")


def normalize_viral_load(data, normalization_type):
    raise FunctionNotImplementedError("""The function to normalize viral load has not been implemented.""")


def forecast_single_instance(data, model_type):
    raise FunctionNotImplementedError("""The function to forecast single instances from data has not been implemented.""")


def detect_seasonality(data):
    raise FunctionNotImplementedError("""The function to detect seasonality in data has not been implemented.""")
