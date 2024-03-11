from weasyprint import HTML
import pandas as pd
import matplotlib.pyplot as plt
from pyWBE.preliminary_functions import (plot_time_series, analyze_trends,
                                         calculate_weekly_concentration_perc_change,
                                         change_point_detection, detect_seasonality,
                                         normalize_viral_load, forecast_single_instance,
                                         get_lead_lag_correlations)


def create_pdf_report(pdf_path: str, time_series_plot: str, trend_plot: str,
                      conc_change_plot: str, change_pt_detect_plot: str,
                      seasonality_plot: str, normalize_plot: str,
                      forecast_plot: str, lead_lag_plot: str,
                      lead_table: pd.DataFrame, lag_table: pd.DataFrame):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=0.75">
    <title>Report</title>
    <style>
        @page {{
            size: letter; /* Set the page size to letter */
            margin: 0.5in; /* Set a 0.5 inch margin */
        }}
        body {{
            width: 6.5in; /* Set body width to 6.5 inches to account for margins */
            margin: 0 auto; /* Center the body within the page */
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
    </style>
    </head>
    <body>
    <h1 style='text-align: center;'>Report</h1>
    <p>The following plot shows how the given time-series data changes over time:</p>
    <img src="{time_series_plot}" alt="Time-series data over time">

    <p>The following plots show the rate of change of the given values as a percentage change and the overall trend of the data:</p>
    <div class="image-container">
        <img src="{conc_change_plot}" alt="Percentage change">
        <img src="{trend_plot}" alt="Overall trend">
    </div>

    <p>The following plot shows the change points in the data i.e., the places where the behavior of the time-series changes:</p>
    <img src="{change_pt_detect_plot}" alt="Change points in the data">

    <p>The given time-series data has the following seasonal components:</p>
    <img src="{seasonality_plot}" alt="Seasonal components">

    <p>And this plot shows the normalized values of the time-series data based on the values in the indicated column:</p>
    <img src="{normalize_plot}" alt="Normalized values of the time-series">

    <p>The single-step forecast from the time-series data looks like this:</p>
    <img src="{forecast_plot}" alt="Single-step forecast">

    <p>The plot below shows the comparison between the two-given time-series data:</p>
    <img src="{lead_lag_plot}" alt="Comparison between two time-series data">

    <p>These tables show the lead and lag correlations between the two time-series data:</p>
    <div class="table-container">
        <table>
            <tr>
                <th>Offset</th>
                <th>Lead Correlation</th>
            </tr>
            <tr>
                <td>1</td>
                <td>0.2</td>
            </tr>
            <tr>
                <td>2</td>
                <td>0.7</td>
            </tr>
            <tr>
                <td>3</td>
                <td>0.01</td>
            </tr>
        </table>

        <table>
            <tr>
                <th>Offset</th>
                <th>Lag Correlation</th>
            </tr>
            <tr>
                <td>1</td>
                <td>0.5</td>
            </tr>
            <tr>
                <td>2</td>
                <td>0.1</td>
            </tr>
            <tr>
                <td>3</td>
                <td>0.01</td>
            </tr>
        </table>
    </div>
    </body>
    </html>
    """
    HTML(string=html_content).write_pdf(pdf_path)


def plot_trends(data: pd.Series, trend_plot_pth: str):
    trends = analyze_trends(data)
    plt.figure(figsize=(15, 8))
    plt.plot(x=data.index, y=data, label='Time-Series Data')
    plt.plot(x=data.index, y=trends, label='Trend')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(trend_plot_pth)
    plt.close()


def plot_conc_change(data: pd.Series, conc_change_plot_pth: str):
    conc_change = calculate_weekly_concentration_perc_change(data)
    plt.figure(figsize=(15, 8))
    plt.plot(x=data.index, y=conc_change, label='Weekly Concentration % Change')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(conc_change_plot_pth)
    plt.close()


def plot_change_pt_detect(data: pd.Series, change_pt_detect_plot_pth: str,
                          model: str = "l2", min_size: int = 28, penalty: int = 1):
    change_points = change_point_detection(data, model, min_size, penalty)
    plt.figure(figsize=(15, 8))
    plt.plot(x=data.index, y=data, label='Time-Series Data')
    plt.plot(x=change_points.index, y=change_points, label='Change Points')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(change_pt_detect_plot_pth)
    plt.close()


def plot_seasonality(data: pd.Series, seasonality_plot_pth: str, model_type: str = "additive"):
    seasonal = detect_seasonality(data, model_type)
    fig, ax = plt.subplots(3, figsize=(15, 8))
    ax[0].plot(x=data.index, y=seasonal.trend)
    ax[0].set_title('Trend')
    ax[1].plot(x=data.index, y=seasonal.seasonal)
    ax[1].set_title('Seasonal')
    ax[2].scatter(x=data.index, y=seasonal.resid)
    ax[2].set_title('Residual')
    ax[2].set_xlabel('Date')
    plt.legend()
    plt.savefig(seasonality_plot_pth)
    plt.close()


def plot_normalize(data: pd.DataFrame, normalize_plot_pth: str, to_normalize: str, normalize_by: str):
    normalized_data = normalize_viral_load(data, to_normalize, normalize_by)
    plt.figure(figsize=(15, 8))
    plt.plot(x=data.index, y=normalized_data, label='Normalized Viral Load')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(normalize_plot_pth)
    plt.close()


def plot_forecast(data: pd.Series, forecast_plot_pth: str, window: pd.DatetimeIndex):
    forecast = forecast_single_instance(data, window)
    plt.figure(figsize=(15, 8))
    plt.plot(x=forecast.index, y=forecast, label='Forecast')
    plt.axvline(x=len(data), color='r', linestyle='--', label='Forecast Start')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
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
    value_series = data_1[value_col]
    time_series = data_1[time_col]
    plot_time_series(time_series, value_series, time_series_plot_pth, plot_type)
    plot_trends(value_series, trend_plot_pth)
    plot_conc_change(value_series, conc_change_plot_pth)
    plot_change_pt_detect(value_series, change_pt_detect_plot_pth, change_pt_model,
                          change_pt_min_size, change_pt_penalty)
    plot_seasonality(value_series, seasonality_plot_pth, seasonality_model)
    plot_normalize(data_1, normalize_plot_pth, value_col, normalize_using_col)
    plot_forecast(value_series, forecast_plot_pth, forecast_window)
    lead_corr, lag_corr = get_lead_lag_correlations(value_series, data_2,
                                                    lead_lag_time_instances, lead_lag_plot_pth,
                                                    lead_lag_max_lag)

    create_pdf_report(pdf_path, time_series_plot_pth, trend_plot_pth, conc_change_plot_pth,
                      change_pt_detect_plot_pth, seasonality_plot_pth, normalize_plot_pth,
                      forecast_plot_pth, lead_lag_plot_pth, lead_corr, lag_corr)
