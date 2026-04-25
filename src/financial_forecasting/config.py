from dataclasses import dataclass, field, replace
from datetime import date, timedelta


SEED = 42


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    ticker: str
    start: str
    end: str
    lag: int
    xgb_params: dict
    lstm_params: dict
    features: list[str] = field(default_factory=lambda: ["Close"])
    train_ratio: float = 0.81
    val_ratio: float = 0.09
    test_ratio: float = 0.10
    uses_technical_indicators: bool = False


def current_end_date() -> str:
    """Return tomorrow so yfinance's exclusive end includes the latest day."""
    return (date.today() + timedelta(days=1)).isoformat()


def materialize_config(config: ExperimentConfig, live: bool = False) -> ExperimentConfig:
    if not live:
        return config
    return replace(config, end=current_end_date())


OIL_CONFIG = ExperimentConfig(
    name="oil",
    ticker="CL=F",
    start="1983-03-31",
    end="2024-06-01",
    lag=5,
    xgb_params={
        "objective": "reg:squarederror",
        "n_estimators": 550,
        "learning_rate": 0.03,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": SEED,
    },
    lstm_params={
        "lstm_units": [40],
        "dense_units": [20],
        "dropout": 0.3,
        "learning_rate": 0.005,
        "l2": 0.005,
        "epochs": 30,
        "batch_size": 64,
        "patience": 5,
        "use_batch_norm": True,
    },
)


TESLA_CONFIG = ExperimentConfig(
    name="tesla",
    ticker="TSLA",
    start="2010-06-29",
    end="2025-01-01",
    lag=5,
    xgb_params={
        "objective": "reg:squarederror",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": SEED,
    },
    lstm_params={
        "lstm_units": [40],
        "dense_units": [20],
        "dropout": 0.3,
        "learning_rate": 0.0005,
        "l2": 0.005,
        "epochs": 100,
        "batch_size": 64,
        "patience": 15,
        "use_batch_norm": True,
    },
)


BITCOIN_CONFIG = ExperimentConfig(
    name="bitcoin",
    ticker="BTC-USD",
    start="2010-01-31",
    end="2024-03-31",
    lag=30,
    features=[
        "Close",
        "EMA_50",
        "Momentum_3",
        "Momentum_7",
        "Rolling_STD_7",
        "Rolling_STD_30",
        "EMA_50_Trend",
        "Price_Trend_3",
        "Price_Trend_7",
    ],
    uses_technical_indicators=True,
    xgb_params={
        "objective": "reg:squarederror",
        "n_estimators": 600,
        "learning_rate": 0.005,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": SEED,
    },
    lstm_params={
        "lstm_units": [160, 120],
        "dense_units": [64],
        "dropout": 0.1,
        "learning_rate": 0.002,
        "l2": 0.001,
        "epochs": 120,
        "batch_size": 32,
        "patience": 15,
        "use_batch_norm": False,
    },
)
