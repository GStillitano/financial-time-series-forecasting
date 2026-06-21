"""Module for downloading and processing financial data."""

import pandas as pd
import ta
import yfinance as yf

from financial_forecasting.config import ExperimentConfig


def download_series(config: ExperimentConfig) -> pd.DataFrame:
    """
    Download financial time series data and add technical indicators if configured.
    """
    # Download data from yfinance without progress bar
    data = yf.download(config.ticker, start=config.start, end=config.end, progress=False)
    data = data.reset_index()
    data["Date"] = pd.to_datetime(data["Date"])
    
    # Sort by date, remove missing values, and reset index
    data = data.sort_values("Date").dropna().reset_index(drop=True)

    # Conditionally add technical indicators based on configuration
    if config.uses_technical_indicators:
        data = add_technical_indicators(data)

    return data


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and add various technical indicators to the DataFrame.
    """
    frame = data.copy()
    close_series = frame["Close"].squeeze()
    
    # Calculate indicators: EMA, Momentum, Rolling STD, and Trends
    frame["EMA_50"] = ta.trend.EMAIndicator(close=close_series, window=50).ema_indicator()
    frame["Momentum_3"] = frame["Close"] - frame["Close"].shift(3)
    frame["Momentum_7"] = frame["Close"] - frame["Close"].shift(7)
    frame["Rolling_STD_7"] = frame["Close"].rolling(window=7).std()
    frame["Rolling_STD_30"] = frame["Close"].rolling(window=30).std()
    frame["EMA_50_Trend"] = frame["EMA_50"] - frame["EMA_50"].shift(3)
    frame["Price_Trend_3"] = frame["Close"] - frame["Close"].shift(3)
    frame["Price_Trend_7"] = frame["Close"] - frame["Close"].shift(7)
    
    # Drop rows with NaN values resulting from rolling/shift operations
    frame = frame.dropna().reset_index(drop=True)
    return frame

