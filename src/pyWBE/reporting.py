import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pyWBE.preliminary_functions import (plot_time_series, analyze_trends,
                                         calculate_weekly_concentration_perc_change,
                                         change_point_detection, detect_seasonality,
                                         normalize_viral_load, forecast_single_instance,
                                         get_lead_lag_correlations)
import ruptures as rpt
import pypandoc


def get_html_table(df: pd.DataFrame):
    """
    This function takes a pandas DataFrame and returns an HTML table. \n
    :param df: pandas DataFrame. \n
    : type df: pd.DataFrame. \n
    :return: str. \n
    """
    html_table = df.to_html(index=False, justify='center')
    return html_table


def create_doc_report(doc_path: str, time_series_plot: str, trend_plot: str,
                      conc_change_plot: str, change_pt_detect_plot: str,
                      seasonality_plot: str, normalize_plot: str,
                      forecast_plot: str, lead_lag_plot: str,
                      lead_table: pd.DataFrame, lag_table: pd.DataFrame):
    """
    This function takes the paths to the plots and tables and creates a DOCX report. \n
    :param doc_path: Path to save the DOCX report. \n
    :type doc_path: str. \n
    :param time_series_plot: Path to the time-series plot. \n
    :type time_series_plot: str. \n
    :param trend_plot: Path to the trend plot. \n
    :type trend_plot: str. \n
    :param conc_change_plot: Path to the concentration change plot. \n
    :type conc_change_plot: str. \n
    :param change_pt_detect_plot: Path to the change point detection plot. \n
    :type change_pt_detect_plot: str. \n
    :param seasonality_plot: Path to the seasonality plot. \n
    :type seasonality_plot: str. \n
    :param normalize_plot: Path to the normalized plot. \n
    :type normalize_plot: str. \n
    :param forecast_plot: Path to the forecast plot. \n
    :type forecast_plot: str. \n
    :param lead_lag_plot: Path to the lead-lag plot. \n
    :type lead_lag_plot: str. \n
    :param lead_table: Lead correlations table. \n
    :type lead_table: pd.DataFrame. \n
    :param lag_table: Lag correlations table. \n
    :type lag_table: pd.DataFrame. \n
    """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        @page {{
            size: letter; /* Set the page size to letter */
            margin: 0.5in; /* Set a 0.5 inch margin */
        }}
        body {{
            width: 6.5in; /* Set body width to 6.5 inches to account for margins */
            margin: 0 auto; /* Center the body within the page */
            margin-bottom: 0.5in; /* Add a 0.5 inch margin to the bottom */
        }}
        /* Flex container for tables */
        .table-container {{
            display: flex;
            justify-content: space-around; /* This will space out the tables evenly */
        }}
        table {{
            border-collapse: collapse; /* Collapse borders so they don't double up */
        }}
        th, td {{
            border: 1px solid black; /* Add borders to table headers and cells */
            text-align: left; /* Align text to the left, adjust as needed */
            padding: 8px; /* Add some padding for content inside cells */
        }}
        /* Styles for centering and scaling images */
        img {{
            max-width: 100%; /* Scale image to fit its container */
            height: auto; /* Maintain aspect ratio */
        }}
        /* Flex container for images */
        .image-container {{
            display: flex;
            justify-content: center; /* Center images within the container */
            gap: 20px; /* Space between images */
            flex-wrap: wrap; /* Allow the images to wrap onto the next line if not enough space */
        }}
        .image-container img {{
            flex: 1; /* Allows the image to grow and fill the container */
            max-width: calc(50% - 10px); /* Adjusts max-width to account for gap, preventing overflow */
        }}
        .text-justify {{
            text-align: justify; /* Justify text */
        }}
    </style>
    </head>
    <body>
    <h1 style='text-align: center;'>Wastewater-Based Epidemiology/COVID-19 Report</h1>
    <p class="text-justify">This report provides a general statistical analysis of the given time-series data, such as trend and seasonality analysis,
    change point analysis, and single step ML-based forecasting.
    </p>
    <h2 style='text-align: center;'>Time-Series Visualization</h2>
    <p>The following plot shows how the given time-series data changes over time:</p>
    <img src="{time_series_plot}" alt="Time-series data over time">

    <h2 style='text-align: center;'>Weekly Percentage Concentration Change</h2>
    <p>The following plot shows the rate of change of the given values as a weekly percentage change:</p>
    <!--
    <div class="image-container">
        <img src="{conc_change_plot}" alt="Percentage change">
        <img src="{trend_plot}" alt="Overall trend">
    </div>
    -->
    <img src="{conc_change_plot}" alt="Percentage change">
    <h2 style='text-align: center;'>Trend Analysis</h2>
    <p>The following plot shows the linear trend followed by the given time-series data:</p>
    <img src="{trend_plot}" alt="Overall trend">

    <h2 style='text-align: center;'>Change Point Detection</h2>
    <p class="text-justify">The following plot shows the change points in the data i.e., the places where the behavior of the time-series changes.
    Change point detection is a statistical technique used to identify points in a time series where the statistical properties
    of the data significantly change. We are using the Pelt algorithm that finds the optimal segmentation by minimizing a cost
    function, such as L1 loss, L2 loss, or the Radial Basis Function (RBF) loss.</p>
    <img src="{change_pt_detect_plot}" alt="Change points in the data">

    <h2 style='text-align: center;'>Seasonality Analysis</h2>
    <p>The given time-series data has the following seasonal components:</p>
    <p class="text-justify">
    <b> Trend Component: </b> The trend component captures the long-term behavior or direction of the time series, ignoring short-term fluctuations
    and seasonal variations. It represents the underlying growth or decline in the data over time, providing insights into the overall trajectory of
    the series. Identifying the trend component is essential for understanding the underlying dynamics and making predictions about future behavior.
    </p>
    <p class="text-justify">
    <b> Seasonal Component: </b> The seasonal component represents the recurring patterns or cycles within the time series that occur at fixed
    intervals, such as daily, weekly, or yearly fluctuations. These patterns often reflect seasonal variations in the data.
    </p>
    <p class="text-justify">
    <b> Residual Component: </b> The residual component, also known as the irregular or noise component, represents the random fluctuations or
    variability in the time series that cannot be explained by the seasonal and trend components. It captures the deviations of the observed data
    from the fitted seasonal and trend patterns, often containing information about random shocks, measurement errors, or other unmodeled factors.
    </p>
    <img src="{seasonality_plot}" alt="Seasonal components">

    <h2 style='text-align: center;'>Normalized Time-Series Plot</h2>
    <p class="text-justify"> And this plot shows the normalized values of the time-series data based on the mean of the values in the indicated column:</p>
    <img src="{normalize_plot}" alt="Normalized values of the time-series">

    <h2 style='text-align: center;'>Single-Step Time-Series Forecast</h2>
    <p class="text-justify"> Currently, we are using a linear regression model to predict the value of the next time step, given the value at the
    current time step. The single-step forecast from the time-series data looks like this:</p>
    <img src="{forecast_plot}" alt="Single-step forecast">

    <h2 style='text-align: center;'>Lead-Lag Correlation Analysis</h2>
    <p class="text-justify">The plot below shows the comparison between the two-given time-series data:</p>
    <img src="{lead_lag_plot}" alt="Comparison between two time-series data">


    <p>These tables show the lead and lag correlations between the two time-series data. The numerical value in the lead/lag column indicates the
    time steps the second times-series data was shifted by and the values column shows the Spearman correlation value for the corresponding shift.
    </p>
    <div class="table-container">
        <div>
            <p>Lead Correlations Table:</p>
            {get_html_table(lead_table)}
        </div>
        <div>
            <p>Lag Correlations Table:</p>
            {get_html_table(lag_table)}
        </div>
    </div>
    </body>
    </html>
    """
    with open(f'{doc_path[:-5]}.html', 'w') as f:
        f.write(html_content)
    pypandoc.convert_file(f'{doc_path[:-5]}.html', 'docx', outputfile=doc_path)
    return None


def plot_trends(data: pd.Series, trend_plot_pth: str):
    """
    This function takes a time-series data, plots the trend of the data and saves it. \n
    :param data: Time-series data. \n
    :type data: pd.Series. \n
    :param trend_plot_pth: Path to save the trend plot. \n
    :type trend_plot_pth: str. \n
    """
    trends = analyze_trends(data)
    plt.figure(figsize=(15, 8))
    plt.plot(data.index, data, "--g", label='Time-Series Data')
    plt.plot(data.index, trends, "r", label='Trend')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Dates', fontsize=16)
    plt.ylabel(data.name, fontsize=16)
    plt.title("Trend of Time-Series Data", fontsize=20)
    plt.legend(fontsize=14)
    plt.savefig(trend_plot_pth)
    plt.close()


def plot_conc_change(data: pd.Series, conc_change_plot_pth: str):
    """
    This function calculates the weekly concentration percentage change, plots it and saves it. \n
    :param data: Time-series data. \n
    :type data: pd.Series. \n
    :param conc_change_plot_pth: Path to save the weekly concentration percentage change plot. \n
    :type conc_change_plot_pth: str. \n
    """
    conc_change = calculate_weekly_concentration_perc_change(data)
    plt.figure(figsize=(15, 8))
    plt.plot(conc_change.index, conc_change, "-ob", label='Weekly Concentration % Change')
    plt.xlabel('Dates', fontsize=16)
    plt.ylabel(data.name, fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.title("Weekly Percentage Concentration Change in Data", fontsize=20)
    plt.legend(fontsize=14)
    plt.savefig(conc_change_plot_pth)
    plt.close()


def plot_change_pt_detect(data: pd.Series, change_pt_detect_plot_pth: str,
                          model: str = "l2", min_size: int = 28, penalty: int = 1):
    """
    This function detects change points in the time-series data, plots it and saves it. \n
    :param data: Time-series data. \n
    :type data: pd.Series. \n
    :param change_pt_detect_plot_pth: Path to save the change point detection plot. \n
    :type change_pt_detect_plot_pth: str. \n
    :param model: Change point detection model. \n
    :type model: str. \n
    :param min_size: The minimum separation (time steps) between
    two consecutive change points detected by the model.\n
    :type min_size: int. \n
    :param penalty: The penalty to be used in the model. \n
    :type penalty: int. \n
    """
    change_points = change_point_detection(data, model, min_size, penalty)
    _, ax = rpt.display(signal=data.values, true_chg_pts=[], computed_chg_pts=change_points, figsize=(15, 8))
    ax[0].xaxis.set_major_locator(MaxNLocator(nbins=10))  # Set maximum number of xticks to 10
    ax[0].set_xticks(range(len(data.index))[::len(data.index)//10])  # Set xticks to every 10th index
    ax[0].set_xticklabels(data.index.date[::len(data.index)//10])  # Set xtick labels to date, skipping some to avoid crowding
    ax[0].set_xlabel('Dates', fontsize=16)
    ax[0].set_ylabel(data.name, fontsize=16)
    plt.title("Change Points in Time-Series Data", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(change_pt_detect_plot_pth)
    plt.close()


def plot_seasonality(data: pd.Series, seasonality_plot_pth: str, model_type: str = "additive"):
    """
    This function detects the seasonality in the time-series data, plots it and saves it. \n
    :param data: Time-series data. \n
    :type data: pd.Series. \n
    :param seasonality_plot_pth: Path to save the seasonality plot. \n
    :type seasonality_plot_pth: str. \n
    :param model_type: Seasonality model type. \n
    :type model_type: str. \n
    """
    seasonal = detect_seasonality(data, model_type)
    _, ax = plt.subplots(3, figsize=(15, 8))
    ax[0].plot(data.index, seasonal.trend, "g")
    ax[0].set_title('Trend', fontsize=20)
    ax[1].plot(data.index, seasonal.seasonal, "g")
    ax[1].set_title('Seasonal', fontsize=20)
    ax[2].scatter(data.index, seasonal.resid, color="g")
    ax[2].set_title('Residual', fontsize=20)
    ax[2].set_xlabel('Date', fontsize=16)
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(seasonality_plot_pth)
    plt.close()


def plot_normalize(data: pd.DataFrame, normalize_plot_pth: str, to_normalize: str, normalize_by: str):
    """
    This function normalizes the time-series data, plots it and saves it. \n
    :param data: Time-series data. \n
    :type data: pd.DataFrame. \n
    :param normalize_plot_pth: Path to save the normalized plot. \n
    :type normalize_plot_pth: str. \n
    :param to_normalize: Column to normalize. \n
    :type to_normalize: str. \n
    :param normalize_by: Column to normalize by. \n
    :type normalize_by: str. \n
    """
    normalized_data = normalize_viral_load(data, to_normalize, normalize_by)
    plt.figure(figsize=(15, 8))
    plt.grid(True)
    plt.plot(data.index, normalized_data, "r", label='Normalized Viral Load')
    plt.xlabel('Dates', fontsize=16)
    plt.ylabel(to_normalize, fontsize=16)
    plt.title("Normalized Values of the Time-Series Data", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize=14)
    plt.savefig(normalize_plot_pth)
    plt.close()


def plot_forecast(data: pd.Series, forecast_plot_pth: str, window: pd.DatetimeIndex):
    """
    This function forecasts the time-series data, plots it and saves it. \n
    :param data: Time-series data. \n
    :type data: pd.Series. \n
    :param forecast_plot_pth: Path to save the forecast plot. \n
    :type forecast_plot_pth: str. \n
    :param window: Forecast window. \n
    :type window: pd.DatetimeIndex. \n
    """
    forecast = forecast_single_instance(data, window)
    window_length = len(window)
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_ylim([forecast.min()-0.5*abs(forecast.min()), forecast.max()+0.5*abs(forecast.max())])
    ax.plot(forecast.index, forecast, label='Time-Series')
    ax.axvspan(window[0], window[-1], alpha=0.3, color='yellow', label='Forecast Window')
    ax.axvline(x=data.index[-1], color='r', linestyle='--', label='Forecast Start')
    inset_ax = plt.axes([0.7, 0.7, 0.25, 0.25])
    inset_ax.plot(forecast.index[-window_length:], forecast.iloc[-window_length:])
    inset_ax.axvline(x=data.index[-1], color='r', linestyle='--', label='Forecast Start')
    inset_ax.set_xticks(forecast.index.date[-window_length::3])
    inset_ax.set_xticklabels(forecast.index.date[-window_length::3], rotation=45)
    inset_ax.set_title('Zoomed-In Forecast')
    ax.set_xlabel('Dates', fontsize=20)
    ax.set_ylabel(data.name, fontsize=20)
    ax.legend(fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_title("Single-Step Forecast of Time-Series Data", fontsize=24)
    # plt.figure(figsize=(15, 8))
    # plt.plot(forecast.index, forecast, label='Forecast')
    # plt.axvline(x=data.index[-2], color='r', linestyle='--', label='Forecast Start')
    # plt.xlabel('Date')
    # plt.ylabel('Value')
    # plt.legend()
    plt.savefig(forecast_plot_pth)
    plt.close()


def generate_report_from_data(data_1: pd.DataFrame, data_2: pd.Series, time_col: str, value_col: str,
                              normalize_using_col: str, forecast_window: pd.DatetimeIndex,
                              lead_lag_time_instances: int, lead_lag_max_lag: int,
                              pdf_path: str, time_series_plot_pth: str, trend_plot_pth: str,
                              conc_change_plot_pth: str, change_pt_detect_plot_pth: str,
                              seasonality_plot_pth: str, normalize_plot_pth: str,
                              forecast_plot_pth: str, lead_lag_plot_pth: str,
                              plot_type: str = "linear", change_pt_model: str = "l2",
                              change_pt_min_size: int = 28, change_pt_penalty: int = 1,
                              seasonality_model: str = "additive"):
    """
    This function generates a PDF report from the given data. \n
    :param data_1: Primary time-series data to perform analysis on. \n
    :type data_1: pd.DataFrame. \n
    :param data_2: Secondary time-series data to be used for correlation analysis. \n
    :type data_2: pd.Series. \n
    :param time_col: Column name indicating dates/time in data_1. \n
    :type time_col: str. \n
    :param value_col: Column name indicating values of interest in data_1. \n
    :type value_col: str. \n
    :param normalize_using_col: Column name to normalize "values" in data_1 by. \n
    :type normalize_using_col: str. \n
    :param forecast_window: Forecast window for the time-series data. \n
    :type forecast_window: pd.DatetimeIndex. \n
    :param lead_lag_time_instances: Number of time instances to consider for lead-lag correlation. \n
    :type lead_lag_time_instances: int. \n
    :param lead_lag_max_lag: Maximum lag to consider for lead-lag correlation. \n
    :type lead_lag_max_lag: int. \n
    :param pdf_path: Path to save the PDF report. \n
    :type pdf_path: str. \n
    :param time_series_plot_pth: Path to save the time-series plot. \n
    :type time_series_plot_pth: str. \n
    :param trend_plot_pth: Path to save the trend plot. \n
    :type trend_plot_pth: str. \n
    :param conc_change_plot_pth: Path to save the concentration change plot. \n
    :type conc_change_plot_pth: str. \n
    :param change_pt_detect_plot_pth: Path to save the change point detection plot. \n
    :type change_pt_detect_plot_pth: str. \n
    :param seasonality_plot_pth: Path to save the seasonality plot. \n
    :type seasonality_plot_pth: str. \n
    :param normalize_plot_pth: Path to save the normalized plot. \n
    :type normalize_plot_pth: str. \n
    :param forecast_plot_pth: Path to save the forecast plot. \n
    :type forecast_plot_pth: str. \n
    :param lead_lag_plot_pth: Path to save the lead-lag plot. \n
    :type lead_lag_plot_pth: str. \n
    :param plot_type: Type of plot to be used for time-series data. \n
    :type plot_type: str. \n
    :param change_pt_model: Change point detection model. \n
    :type change_pt_model: str. \n
    :param change_pt_min_size: The minimum separation (time steps) between
    two consecutive change points detected by the model.\n
    :type change_pt_min_size: int. \n
    :param change_pt_penalty: The penalty to be used in the model. \n
    :type change_pt_penalty: int. \n
    :param seasonality_model: Seasonality model type. \n
    :type seasonality_model: str. \n
    :return: HTML of generated report. \n
    :rtype: str. \n
    """
    value_series = data_1[value_col]
    time_series = data_1[time_col]
    plot_time_series(time_series, value_series, time_series_plot_pth, plot_type)
    lead_corr, lag_corr = get_lead_lag_correlations(value_series, data_2,
                                                    lead_lag_time_instances, lead_lag_plot_pth,
                                                    lead_lag_max_lag)
    plot_trends(value_series, trend_plot_pth)
    plot_conc_change(value_series, conc_change_plot_pth)
    plot_change_pt_detect(value_series, change_pt_detect_plot_pth, change_pt_model,
                          change_pt_min_size, change_pt_penalty)
    plot_seasonality(value_series, seasonality_plot_pth, seasonality_model)
    plot_normalize(data_1, normalize_plot_pth, value_col, normalize_using_col)
    plot_forecast(value_series, forecast_plot_pth, forecast_window)

    html_content = create_doc_report(pdf_path, time_series_plot_pth, trend_plot_pth, conc_change_plot_pth,
                                     change_pt_detect_plot_pth, seasonality_plot_pth, normalize_plot_pth,
                                     forecast_plot_pth, lead_lag_plot_pth, lead_corr, lag_corr)
    return html_content
