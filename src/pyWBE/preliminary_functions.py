"""
=====================
Preliminary Functions
=====================

Contains preliminary functions used to aid data analysis.

Note: Add type-hints and docstrings to functions as they are implemented.

"""


from pyWBE.exceptions import FunctionNotImplementedError


def plot_time_series(series_x, series_y, plot_type):
    raise FunctionNotImplementedError("""The function to plot time series data has not been implemented.""")


def calculate_weekly_concentration_perc_change(dates, conc_data):
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
