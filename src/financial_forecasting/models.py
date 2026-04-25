import random

import numpy as np
import tensorflow as tf
import xgboost as xgb
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input, LSTM
from tensorflow.keras.regularizers import l2

from financial_forecasting.config import SEED


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def fit_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
) -> tuple[xgb.XGBRegressor, dict]:
    model = xgb.XGBRegressor(**params, eval_metric="rmse")
    model.fit(
        X_train,
        y_train.ravel(),
        eval_set=[(X_train, y_train.ravel()), (X_val, y_val.ravel())],
        verbose=False,
    )
    evals_result = model.evals_result() if hasattr(model, "evals_result") else {}
    return model, evals_result


def build_lstm_model(input_shape: tuple[int, int], params: dict) -> Sequential:
    model = Sequential()
    model.add(Input(shape=input_shape))

    lstm_units = params["lstm_units"]
    for index, units in enumerate(lstm_units):
        return_sequences = index < len(lstm_units) - 1
        model.add(
            LSTM(
                units,
                activation="tanh",
                return_sequences=return_sequences,
                kernel_regularizer=l2(params["l2"]),
            )
        )
        if params.get("use_batch_norm") and index == len(lstm_units) - 1:
            model.add(BatchNormalization())
        model.add(Dropout(params["dropout"]))

    for units in params["dense_units"]:
        model.add(Dense(units, activation="tanh", kernel_regularizer=l2(params["l2"])))
        model.add(Dropout(params["dropout"]))

    model.add(Dense(1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="mean_squared_error",
    )
    return model


def fit_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
) -> tuple[Sequential, tf.keras.callbacks.History]:
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]), params)
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=params["patience"],
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        callbacks=callbacks,
        verbose=0,
    )
    return model, history

