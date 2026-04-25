import json
from pathlib import Path

import matplotlib.pyplot as plt

from financial_forecasting.config import BITCOIN_CONFIG, ExperimentConfig
from financial_forecasting.data import download_series
from financial_forecasting.features import flatten_lag_features, prepare_sequences
from financial_forecasting.metrics import regression_summary
from financial_forecasting.models import fit_lstm, fit_xgboost, set_global_seed


def save_training_plot(values_a: list[float], values_b: list[float], title: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(values_a, label="Train Loss")
    plt.plot(values_b, label="Validation Loss")
    plt.title(title)
    plt.xlabel("Epochs / Rounds")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_prediction_plot(dates, y_true, y_pred, title: str, output_path: Path, label: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(dates, y_true, label="Actual", color="blue", alpha=0.7)
    plt.plot(dates, y_pred, label=label, color="green")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_experiment(config: ExperimentConfig, results_dir: str | Path = "results") -> dict:
    set_global_seed()
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    data = download_series(config)
    bundle = prepare_sequences(data, config)

    X_train_xgb = flatten_lag_features(bundle.X_train)
    X_val_xgb = flatten_lag_features(bundle.X_val)
    X_test_xgb = flatten_lag_features(bundle.X_test)

    xgb_model, xgb_eval = fit_xgboost(
        X_train=X_train_xgb,
        y_train=bundle.y_train,
        X_val=X_val_xgb,
        y_val=bundle.y_val,
        params=config.xgb_params,
    )

    xgb_train_pred = bundle.scaler_y.inverse_transform(
        xgb_model.predict(X_train_xgb).reshape(-1, 1)
    )
    xgb_test_pred = bundle.scaler_y.inverse_transform(
        xgb_model.predict(X_test_xgb).reshape(-1, 1)
    )

    lstm_model, history = fit_lstm(
        X_train=bundle.X_train,
        y_train=bundle.y_train,
        X_val=bundle.X_val,
        y_val=bundle.y_val,
        params=config.lstm_params,
    )

    lstm_train_pred = bundle.scaler_y.inverse_transform(lstm_model.predict(bundle.X_train, verbose=0))
    lstm_test_pred = bundle.scaler_y.inverse_transform(lstm_model.predict(bundle.X_test, verbose=0))

    summary = {
        "experiment": config.name,
        "ticker": config.ticker,
        "xgboost": {
            "train": regression_summary(bundle.y_train_real, xgb_train_pred),
            "test": regression_summary(bundle.y_test_real, xgb_test_pred),
        },
        "lstm": {
            "train": regression_summary(bundle.y_train_real, lstm_train_pred),
            "test": regression_summary(bundle.y_test_real, lstm_test_pred),
        },
    }

    if xgb_eval:
        validation_zero = xgb_eval.get("validation_0", {})
        validation_one = xgb_eval.get("validation_1", {})
        metric_key = "rmse"
        if metric_key in validation_zero and metric_key in validation_one:
            save_training_plot(
                validation_zero[metric_key],
                validation_one[metric_key],
                f"{config.name.title()} XGBoost Training vs Validation Loss",
                results_path / f"{config.name}_xgboost_loss.png",
            )

    save_prediction_plot(
        bundle.date_test,
        bundle.y_test_real,
        xgb_test_pred,
        f"{config.name.title()} Close Price Forecast with XGBoost",
        results_path / f"{config.name}_xgboost_predictions.png",
        "Predicted (XGBoost)",
    )

    save_training_plot(
        history.history["loss"],
        history.history["val_loss"],
        f"{config.name.title()} LSTM Training vs Validation Loss",
        results_path / f"{config.name}_lstm_loss.png",
    )

    save_prediction_plot(
        bundle.date_test,
        bundle.y_test_real,
        lstm_test_pred,
        f"{config.name.title()} Close Price Forecast with LSTM",
        results_path / f"{config.name}_lstm_predictions.png",
        "Predicted (LSTM)",
    )

    summary_path = results_path / f"{config.name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_all(results_dir: str | Path = "results") -> list[dict]:
    from financial_forecasting.config import OIL_CONFIG, TESLA_CONFIG

    configs = [OIL_CONFIG, TESLA_CONFIG, BITCOIN_CONFIG]
    return [run_experiment(config, results_dir=results_dir) for config in configs]

