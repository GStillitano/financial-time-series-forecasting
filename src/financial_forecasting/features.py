"""Module for feature engineering and sequence generation for ML models."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from financial_forecasting.config import ExperimentConfig


@dataclass
class DatasetBundle:
    """Dataclass holding all subsets of a split dataset and their scalers."""
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    y_train_real: np.ndarray
    y_val_real: np.ndarray
    y_test_real: np.ndarray
    date_train: np.ndarray
    date_val: np.ndarray
    date_test: np.ndarray
    scaler_X: MinMaxScaler | None
    scaler_y: MinMaxScaler


def prepare_sequences(data: pd.DataFrame, config: ExperimentConfig) -> DatasetBundle:
    """
    Prepare sequential features and targets, splitting into train, validation, and test sets.
    """
    n_total = len(data)
    # Define split indices
    train_end = int(n_total * config.train_ratio)
    val_end = int(n_total * (config.train_ratio + config.val_ratio))

    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))

    # Fit scalers only on training data to prevent data leakage
    scaler_X.fit(data.loc[: train_end - 1, config.features])
    scaler_y.fit(data.loc[: train_end - 1, ["Close"]])

    X_all = scaler_X.transform(data[config.features])
    y_all = scaler_y.transform(data[["Close"]])
    dates = pd.to_datetime(data["Date"]).to_numpy()

    # Create sequences using the specified lag
    X_seq = []
    y_seq = []
    y_dates = []
    for index in range(len(data) - config.lag):
        X_seq.append(X_all[index : index + config.lag])
        y_seq.append(y_all[index + config.lag])
        y_dates.append(dates[index + config.lag])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq).reshape(-1, 1)
    y_dates = np.array(y_dates)

    # Calculate indices for sequences
    target_train_end = max(train_end - config.lag, 1)
    target_val_end = max(val_end - config.lag, target_train_end + 1)

    # Split sequences into train, validation, and test
    X_train = X_seq[:target_train_end]
    X_val = X_seq[target_train_end:target_val_end]
    X_test = X_seq[target_val_end:]

    y_train = y_seq[:target_train_end]
    y_val = y_seq[target_train_end:target_val_end]
    y_test = y_seq[target_val_end:]

    date_train = y_dates[:target_train_end]
    date_val = y_dates[target_train_end:target_val_end]
    date_test = y_dates[target_val_end:]

    return DatasetBundle(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        y_train_real=scaler_y.inverse_transform(y_train),
        y_val_real=scaler_y.inverse_transform(y_val),
        y_test_real=scaler_y.inverse_transform(y_test),
        date_train=date_train,
        date_val=date_val,
        date_test=date_test,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
    )


def flatten_lag_features(X: np.ndarray) -> np.ndarray:
    """
    Flatten the lag features to 2D for models like XGBoost.
    """
    return X.reshape(X.shape[0], -1)

