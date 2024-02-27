"""
=====================
Preliminary Functions
=====================

Contains preliminary functions used to aid data analysis.

Note: Add type-hints and docstrings to functions as they are implemented.

"""


from pyWBE.exceptions import DurationTooShortError, DurationExceededError
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
import ruptures as rpt
import calmap
import io
import matplotlib.colors as mcolors


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
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return buf
    elif plot_type == "log":
        plt.plot(series_x, np.log(series_y))
        plt.xlabel(series_x.name)
        plt.ylabel(f"{series_y.name} (log)")
        plt.title("Time Series Visualization (log scale)")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return buf
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


def normalize_viral_load(data: pd.DataFrame, to_normalize: str, normalize_by: str) -> pd.Series:
    """
    This function normalizes the time-series data given in
    the "to_normalize" column of the data using the values
    in the "normalize_by" column of the data.\n
    :param data: The Pandas DataFrame containing the relevant data.\n
    :type data: Pandas DataFrame\n
    :param to_normalize: The name of the column containing the data to be normalized.\n
    :type to_normalize: str\n
    :param normalize_by: The name of the column containing the data to normalize by.\n
    :type normalize_by: str\n
    :return: The normalized data.\n
    :rtype: Pandas Series\n
    """
    if to_normalize in data.columns and normalize_by in data.columns:
        return data[to_normalize] / data[normalize_by].mean()
    else:
        raise ValueError(f"The columns {to_normalize} and/or {normalize_by} are not present in the given data.")


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

        one_time_step_pred = model.predict(training_vals[-1].reshape(-1, 1))

        time_diff = data.index[-1] - data.index[-2]
        next_index = data.index[-1] + time_diff
        data[next_index] = one_time_step_pred[0]

        return data


def detect_seasonality(data: pd.Series, model_type: str = "additive") -> pd.DataFrame:
    """
    This function analyzes a given time-series data for seasonality.\n
    :param data: A Pandas Series, assumed to have dates as its indices
    with the corresponding values of the time-series data.\n
    :type data: pd.Series\n
    :param model_type: Can be "additive" or "multiplicative", determines
    the type of seasonality model assumed for the data.\n
    :type model_type: str\n
    :return: Returns a Pandas DataFrame that contain the Trend, Seasonal,
    and Residual components computed using the given model type. Can be
    plotted using the "plot" method of Pandas DataFrame class.\n
    :rtype: pd.DataFrame\n
    """
    decompose_result = seasonal_decompose(data, model=model_type)
    return decompose_result


def get_lead_lag_correlations(x: pd.Series, y: pd.Series, time_instances: int, max_lag: int = 3):
    """
    This function computes the lead and lag correlations between two
    given time-series data.\n
    :param x: The first time-series data.\n
    :type x: pd.Series\n
    :param y: The second time-series data.\n
    :type y: pd.Series\n
    :param time_instances: The number of time instances to be considered
    for the correlation analysis.\n
    :type time_instances: int\n
    :param max_lag: The maximum lag time to be considered for the
    correlation analysis.\n
    :type max_lag: int\n
    :return: Returns the lead and lag correlations between the
    given time-series data and the buffer where the time-series
    comparision is stored.\n
    :rtype: Tuple\n
    """

    x = x.iloc[-time_instances:]
    y = y.iloc[-time_instances:]

    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    buf = io.BytesIO()

    calmap.yearplot(x, ax=ax[0], cmap='YlGn',
                    fillcolor='grey', linewidth=2,
                    daylabels='MTWTFSS', dayticks=[0, 2, 4, 6])
    ax[0].set_title('Time Series 1')
    calmap.yearplot(y, ax=ax[1], cmap='YlGn',
                    fillcolor='grey', linewidth=2,
                    daylabels='MTWTFSS', dayticks=[0, 2, 4, 6])
    ax[1].set_title('Time Series 2')

    dates_str = f"{x.index[0].date()} to {x.index[-1].date()}"
    fig.suptitle(f"Comparision of the two time series data \n from dates {dates_str}", x=0.45, fontsize=16)
    plt.tight_layout()

    cmap = plt.cm.YlGn
    norm = mcolors.Normalize(vmin=min(x.min(), y.min()), vmax=max(x.max(), y.max()))  # Adjust the range if needed
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    # Plot a colorbar legend
    plt.colorbar(scalar_mappable, ax=ax, label='Time-Series Values', location="right")
    plt.savefig(buf, format='png')
    buf.seek(0)

    x, y = x.to_frame(), y.to_frame()
    Is = range(-max_lag, max_lag)
    dfs = pd.DataFrame()

    for i in Is:
        x_shifted = x.shift(i)
        x_shifted['target_class'] = y
        dfs[i] = x_shifted.corr(method='spearman')['target_class']

    dfs_T = dfs.iloc[:-1, :].T
    correlations = pd.DataFrame()
    correlations['Lags'] = dfs_T.idxmax()
    correlations['values'] = dfs_T.max()

    lead_corr = correlations[[correlations['values'] >= 0] and correlations['Lags'] <= 0]  # With only lag time
    lag_corr = correlations[correlations['values'] >= 0]   # with lead and lag time both

    return lead_corr, lag_corr, buf
