# Forecasting Financial Asset Closing Prices with XGBoost and LSTM

## Authors

Davide Aloisi, Edoardo Besteghi, Giuseppe Stillitano

## Abstract

This project studies next-step forecasting for three financial time series drawn from different market environments: crude oil, Tesla stock, and Bitcoin. Two modeling approaches are compared: XGBoost, as a machine learning baseline based on gradient-boosted decision trees, and LSTM, as a deep learning architecture designed for sequential data. The experiments are structured to highlight how model performance changes across assets with different levels of volatility, nonlinearity, and market-specific dynamics. Results show that both methods perform well on crude oil, LSTM clearly outperforms XGBoost on Tesla, and Bitcoin requires richer feature engineering to obtain satisfactory forecasts. Across all three experiments, LSTM provides the most consistent overall performance.

## 1. Introduction

Forecasting financial time series remains a challenging task because price dynamics are shaped by noise, structural breaks, nonlinear dependencies, and exogenous shocks. Even when the prediction target is the same, such as the next closing price, different assets may require different modeling strategies. This project investigates that issue by comparing one machine learning model and one deep learning model across three distinct financial series:

1. Crude oil closing price
2. Tesla stock closing price
3. Bitcoin closing price (`BTC-USD`)

These assets were chosen because they represent different market regimes: commodities, equities, and cryptocurrencies. The core objective is to evaluate whether a simpler supervised learning approach based on lag features can remain competitive, or whether a sequence-oriented neural architecture is better suited to the task.

## 2. Data

All datasets were downloaded with `yfinance`. The original data include the standard market variables `Date`, `Open`, `High`, `Low`, `Close`, and `Volume`. For the crude oil and Tesla experiments, only the closing price was used. For Bitcoin, a richer feature space was introduced after initial experiments suggested that raw lagged prices alone were not sufficient.

### 2.1 Assets and time ranges

| Asset | Ticker | Date range |
| --- | --- | --- |
| Crude oil | `CL=F` | `1983-03-31` to `2024-06-01` |
| Tesla | `TSLA` | `2010-06-29` to `2025-01-01` |
| Bitcoin | `BTC-USD` | `2010-01-31` to `2024-03-31` |

## 3. Methodology

### 3.1 Forecasting setup

The forecasting task is defined as one-step-ahead prediction: given the previous `n` observations, the model predicts the closing price at the next time step. For crude oil and Tesla, the lag window is set to 5 days. For Bitcoin, the lag window is extended to 30 days to capture longer temporal dependencies.

The data are split chronologically to avoid information leakage:

- 81% training
- 9% validation
- 10% test

All input data are scaled to the range `(-1, 1)`. This is particularly convenient for the `tanh` activation used in the LSTM networks and was kept consistently across experiments.

### 3.2 Models

#### XGBoost

XGBoost is used as the main non-neural baseline. Since it does not natively model temporal order, the time series is transformed into a supervised learning dataset through lagged features. The model is trained to regress the next closing price from the lag window.

#### LSTM

LSTM is used to model temporal dependencies directly from sequential inputs. Its gating structure makes it suitable for time series in which past information may remain relevant over longer horizons. In this project, the LSTM models are regularized through dropout, early stopping, and, in some cases, `L2` penalties and batch normalization.

### 3.3 Bitcoin feature engineering

Bitcoin was the most difficult series in the project, so additional technical indicators were introduced to enrich the feature space:

- `EMA_50`
- `Momentum_3`
- `Momentum_7`
- `Rolling_STD_7`
- `Rolling_STD_30`
- `EMA_50_Trend`
- `Price_Trend_3`
- `Price_Trend_7`

Together with the closing price, these variables provide information about trend, short-term momentum, and local volatility.

### 3.4 Evaluation metrics

Model performance is evaluated with:

- Mean Absolute Error (`MAE`)
- Mean Absolute Percentage Error (`MAPE`)
- Coefficient of determination (`R^2`)

In addition to numerical metrics, the project also uses learning-curve plots and actual-versus-predicted plots to inspect model behavior and possible overfitting.

## 4. Experimental Results

### 4.1 Crude oil

| Model | Train MAE | Test MAE | Test MAPE | Test `R^2` |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 1.5197 | 2.2792 | 2.58% | 0.9402 |
| LSTM | 1.4012 | 2.0193 | 2.31% | 0.9499 |

Both models perform strongly on crude oil, suggesting that this series is comparatively easier to forecast in the chosen setup. LSTM achieves slightly better performance on all reported metrics. The main prediction difficulties appear around sharp spikes, which are likely associated with external shocks that are not directly encoded in the historical price signal.

### 4.2 Tesla

| Model | Train MAE | Test MAE | Test MAPE | Test `R^2` |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 4.8444 | 18.3927 | 7.14% | 0.8114 |
| LSTM | 5.0859 | 7.4102 | 3.07% | 0.9690 |

Tesla is substantially more difficult than crude oil, especially for XGBoost. The large gap between training and test error suggests overfitting in the tree-based model. LSTM, by contrast, captures the variability of the series much more effectively and delivers a much stronger `R^2` and lower `MAPE`.

### 4.3 Bitcoin

| Model | Train MAE | Test MAE | Test MAPE | Test `R^2` |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 738.3839 | 2376.6983 | 6.04% | 0.9304 |
| LSTM | 637.0484 | 1095.1069 | 6.04% | 0.9812 |

Bitcoin is the most complex asset in the project. After extending the feature space with technical indicators, both models achieve acceptable performance, but LSTM again remains the most reliable. XGBoost still shows a noticeable increase in error from training to test, which suggests some degree of overfitting. LSTM handles the enriched sequential structure better and achieves the highest `R^2` among all Bitcoin results.

## 5. Discussion

The three experiments suggest that forecasting performance depends strongly on the structure of the underlying asset. Crude oil is relatively well behaved under the chosen setup, and both XGBoost and LSTM produce accurate predictions. Tesla exhibits stronger nonlinearities and variability, which reduce the effectiveness of the tree-based approach but remain manageable for LSTM. Bitcoin is the most demanding case, and its performance improves only after introducing additional domain-informed features.

Two broader observations emerge from the project. First, sequence-aware models appear to generalize more consistently across assets with different dynamics. Second, feature engineering remains important, especially when the raw price alone is insufficient to describe the relevant market structure.

## 6. Conclusion

This project compares XGBoost and LSTM on three financial time series from different markets. The results show that:

- both models perform well on crude oil
- LSTM clearly outperforms XGBoost on Tesla
- Bitcoin benefits from richer engineered features and longer lag windows
- across all three experiments, LSTM is the most consistent model

Overall, the experiments support the idea that financial forecasting tasks that look similar on the surface can require different levels of model complexity. For the three datasets considered here, LSTM is the more robust choice.

## 7. Reproducibility Notes

The repository includes modular Python scripts for downloading data, creating lag-based inputs, training models, computing metrics, and saving plots. Results may vary slightly across runs due to changes in downloaded market data and the stochastic nature of training, even when seeds are fixed.

The original work was first developed as a notebook-based experiment and then reorganized into a repository structure better suited for version control, reproducibility, and future extensions.
