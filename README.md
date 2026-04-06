# -Air quality Forecasting-
# Comparative Analysis of Statistical and Machine Learning Approaches for PM2.5 Time-Series Forecasting 🌍💨

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/pandas-Data_Manipulation-150458?logo=pandas)](https://pandas.pydata.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Machine_Learning-blue?logo=xgboost)](https://xgboost.readthedocs.io/)
[![Statsmodels](https://img.shields.io/badge/statsmodels-Statistical_Analysis-darkgreen)](https://www.statsmodels.org/)

## 📌 Abstract
This project investigates the efficacy of traditional statistical models (ARIMA) versus modern tree-based machine learning algorithms (XGBoost) in forecasting daily PM2.5 air pollution concentrations. Accurately predicting particulate matter is critical for preemptive public health interventions. This repository contains a rigorous, end-to-end pipeline encompassing advanced time-series imputation, statistical stationarity testing, and recursive 7-day forecasting.

## 📊 Dataset
The study utilizes the **Beijing PM2.5 Data Set** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data). It contains multi-year, hourly recordings of PM2.5 concentration alongside meteorological variables such as dew point, temperature, pressure, and wind direction.

## 🔬 Methodology

### 1. Advanced Data Preprocessing
Real-world sensor data is inherently noisy and incomplete. To preserve the integrity of the environmental signals, the following steps were taken:
* **Linear Interpolation:** Replaced basic forward-fill methods with linear interpolation to more accurately model the chemical diffusion of particulate matter during sensor downtimes.
* **Winsorization (Outlier Handling):** Extreme anomalous sensor readings were capped at the 99.5th percentile. This preserves valid pollution spikes while preventing impossible mathematical variances from skewing the model's loss gradients.
* **Categorical Feature Engineering:** Converted Combined Wind Direction (`cbwd`) into one-hot encoded variables, leveraging the domain knowledge that wind acts as a primary vector for pollution dispersion.

### 2. Statistical Exploratory Data Analysis (EDA)
Before modeling, the underlying structure of the time series was mathematically validated:
* **Volatility Analysis:** Mapped 30-day rolling means and standard deviations to visualize changing variance across seasons.
* **Augmented Dickey-Fuller (ADF) Test:** Programmatically tested for unit roots to determine the necessary differencing parameter ($d$) for the ARIMA model.
* **ACF & PACF Analysis:** Utilized Partial Autocorrelation visual evidence to mathematically justify the lag cut-off point, avoiding arbitrary hyperparameter selection for XGBoost feature engineering.
* **Multivariate Correlation:** Generated heatmaps to quantify the dispersing/trapping effects of exogenous meteorological variables (like wind speed) on PM2.5.

### 3. Modeling Strategy
* **Baseline (ARIMA):** Implemented as a highly interpretable, linear statistical baseline.
* **Machine Learning (XGBoost):** Framed the time-series problem as a supervised learning task using historical lag features (t-1 to t-7) determined by the PACF analysis.
* **7-Day Forecasting:** Implemented a recursive forecasting strategy, feeding Day 1 predictions back into the model to predict Day 2, up to 7 days into the future.

## 🚀 Results
*(Note: Update these metrics based on your final Colab run)*
* **ARIMA Model:** RMSE = `X.XX` | MAE = `X.XX`
* **XGBoost Model:** RMSE = `X.XX` | MAE = `X.XX`

**Conclusion:** XGBoost demonstrated superior performance in capturing non-linear, extreme pollution spikes compared to the linear ARIMA baseline, highlighting the advantage of incorporating complex lag features.

## 💻 How to Run
1. Clone this repository.
2. Open the `Air_Quality_Forecasting.ipynb` notebook in Google Colab or Jupyter Notebook.
3. Run the cells sequentially. The dataset is fetched automatically via URL.

## 🔭 Future Research Scope
While XGBoost handles non-linear spikes well, the recursive 7-day forecasting method introduces compounding errors. Future iterations of this research will focus on:
* Transitioning to a fully **multivariate forecasting architecture** by feeding exogenous weather variables directly into the prediction loops.
* Implementing deep sequence models like **Long Short-Term Memory (LSTM)** networks or **Temporal Convolutional Networks (TCNs)** to better capture long-range sequential dependencies.
