# Financial Time Series Forecasting

Forecasting the closing prices of three financial assets with machine learning and deep learning models:

- Crude oil (`CL=F`)
- Tesla (`TSLA`)
- Bitcoin (`BTC-USD`)

The project compares `XGBoost` and `LSTM` on time series with very different market dynamics in order to study how model behavior changes across energy, equities, and cryptocurrencies.

## Objective

The goal is to predict the next closing price from historical data and evaluate whether a tree-based machine learning model or a recurrent neural network is better suited to each series.

The project focuses on:

- time series forecasting with lag-based inputs
- comparison between classical ML and deep learning
- performance differences across assets with different volatility and structure
- the effect of feature engineering on more complex series such as Bitcoin

## Datasets

Data are downloaded with `yfinance`.

- Oil: closing prices from `1983-03-31` to `2024-06-01`
- Tesla: closing prices from `2010-06-29` to `2025-01-01`
- Bitcoin: closing prices from `2010-01-31` to `2024-03-31`

For oil and Tesla, the experiments use the closing price as the main signal.  
For Bitcoin, additional technical indicators are included to better model the complexity of the series.

## Methods

### XGBoost

XGBoost is used as a strong machine learning baseline. Since it is not inherently sequential, the time series is transformed into supervised learning samples through lag features.

### LSTM

LSTM is used as the deep learning model for capturing temporal dependencies in sequential data. It is particularly useful when the dynamics of the series are more complex or nonlinear.

## Features

### Oil and Tesla

- closing price only
- lag window of 5 days

### Bitcoin

- closing price
- `EMA_50`
- `Momentum_3`
- `Momentum_7`
- `Rolling_STD_7`
- `Rolling_STD_30`
- `EMA_50_Trend`
- `Price_Trend_3`
- `Price_Trend_7`
- lag window of 30 days

## Evaluation

The models are evaluated with:

- `MAE`
- `MAPE`
- `R^2`

The dataset split follows:

- 81% training
- 9% validation
- 10% test

The repository also saves:

- training vs validation loss plots
- actual vs predicted value plots
- JSON summaries of the results

## Project Structure

```text
financial-time-series-forecasting/
├── report/
│   └── report.md
├── results/
├── scripts/
│   ├── run_all.py
│   ├── run_bitcoin.py
│   ├── run_oil.py
│   └── run_tesla.py
├── src/
│   └── financial_forecasting/
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── metrics.py
│       ├── models.py
│       └── pipeline.py
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Run a single experiment:

```bash
PYTHONPATH=src python scripts/run_oil.py
PYTHONPATH=src python scripts/run_tesla.py
PYTHONPATH=src python scripts/run_bitcoin.py
```

Run all experiments:

```bash
PYTHONPATH=src python scripts/run_all.py
```

Outputs are saved in `results/`.

The default scripts keep the fixed date ranges used for the research experiments. If you want a future-oriented run that uses the current date as the data endpoint, you can use:

```bash
PYTHONPATH=src python scripts/run_all.py --live
```

The `--live` flag updates only the end date, so the default fixed-date baseline remains unchanged.

## Main Findings

- Oil is the easiest series among the three, and both models perform well.
- Tesla is more challenging, and LSTM clearly outperforms XGBoost.
- Bitcoin is the most complex series, and additional technical indicators improve forecasting performance.
- Across the three experiments, LSTM provides the most consistent results.

## Report

The full project write-up is available in [report/report.md](/Users/giusestilly/Documents/Codex/2026-04-25/files-mentioned-by-the-user-progetto/financial-time-series-forecasting/report/report.md).

## Citation

Citation metadata is provided in [CITATION.cff](/Users/giusestilly/Documents/Codex/2026-04-25/files-mentioned-by-the-user-progetto/financial-time-series-forecasting/CITATION.cff).
