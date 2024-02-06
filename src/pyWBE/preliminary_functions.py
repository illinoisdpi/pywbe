"""
=====================
Preliminary Functions
=====================

Contains preliminary functions used to aid data analysis.

Note: Add type-hints and docstrings to functions as they are implemented.

"""


from pyWBE.exceptions import FunctionNotImplementedError
from pyWBE.exceptions import DurationTooShortError, DurationExceededError
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


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


def analyze_trends(data, analysis_type):
    raise FunctionNotImplementedError("""The function to analyze trends has not been implemented.""")


def change_point_detection(time_instance, method):
    raise FunctionNotImplementedError("""The function to change point detection has not been implemented.""")


def normalize_viral_load(data, normalization_type):
    raise FunctionNotImplementedError("""The function to normalize viral load has not been implemented.""")


def forecast_single_instance(data: pd.Series, window: pd.DatetimeIndex) -> pd.Series:
    """
    This function predicts the value of the given time-series data
    a single time-step into the future using a Linear Regression
    model trained on the data specified by the parameter "window_length".\n
    :param data: A Pandas Series, assumed to have dates as its indices,
    containing the time-series data whose value needs to be predicted
    in the future.\n
    :type data: pd.Series\n
    :param window: A Pandas DateTimeIndex containing date range for
    the "data" that must be used to train the Linear Regression model.
    Minimum length must be 1 week and maximum length can be the entire
    date range of the "data".\n
    :type window: pd.DateTimeIndex\n
    :return: Returns the original "data" with the next time-step
    prediction appended to it.\n
    :rtype: pd.Series\n
    """
    one_week = pd.Timedelta(days=7)
    window_duration = window.max() - window.min()
    data_duration = data.index.max() - data.index.min()

    if window_duration < one_week:
        raise DurationTooShortError("""Window length is too short. Should be atleast 1 week.""")
    elif window_duration > data_duration:
        raise DurationExceededError("""Window length is too long. Should not exceed given data's duration.""")
    else:
        training_vals = data[data.index.isin(window)].to_numpy().reshape(-1, 1)
        X_train = training_vals[:-1]
        y_train = training_vals[1:]

        model = LinearRegression()
        model.fit(X_train, y_train)

        one_time_step_pred = model.predict(training_vals[-1])

        time_diff = data.index[-1] - data.index[-2]
        next_index = data.index[-1] + time_diff
        data[next_index] = one_time_step_pred

        return data


def detect_seasonality(data):
    raise FunctionNotImplementedError("""The function to detect seasonality in data has not been implemented.""")
