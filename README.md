# Beverage Sales Forecast API

> **Python version: 3.13 required.** Python 3.14 breaks TensorFlow. Use 3.13 for local training and model development.

A time series forecasting system that trains multiple models, automatically selects the best performer per state, and serves 8 week beverage sales forecasts via a REST API.

The API serves precomputed forecasts from JSON.



## Project Structure

```
forecasting/
├── data/
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── xgboost_model.py
│   │   ├── prophet_model.py
│   │   ├── arima_model.py
│   │   ├── lstm_model.py
│   │   └── forecast.py
│   └── api/
│       └── main.py
├── models/                         
├── forecasts.json                  
├── tournament_results_v2.json      
├── xgb_feature_columns.json        
├── config.py
├── requirements.txt
└── README.md
```



## Dataset

- Weekly beverage sales across 43 US states
- Date range: January 2019 - December 2023
- 226 data points per state after preprocessing
- Single category: Beverages


## Setup & Installation

```bash
git clone https://github.com/Kushal2205a/beverage-sales-api.git
cd beverage-sales-api
pip install -r requirements.txt
```

To run the API locally:

```bash
python -m uvicorn forecasting.src.api.main:app --reload                                 
```

Then open `http://127.0.0.1:8000/docs` for the interactive Swagger UI.



## API Endpoints

## `GET /forecast/{state}`
Returns 8-week forecast for the specified state using the best-performing model.

```
GET /forecast/california
```

```json
{
  "state": "California",
  "model_used": "Prophet",
  "forecast_horizon_weeks": 8,
  "forecasts": [
    { "date": "2023-12-10", "predicted_sales": 1076453710.9 },
    { "date": "2023-12-17", "predicted_sales": 1042187558.24 },
    { "date": "2023-12-24", "predicted_sales": 1203836959.85 },
    { "date": "2023-12-31", "predicted_sales": 1429713441.47 },
    { "date": "2024-01-07", "predicted_sales": 1525029556.62 },
    { "date": "2024-01-14", "predicted_sales": 1370812239.41 },
    { "date": "2024-01-21", "predicted_sales": 1154205187.7  },
    { "date": "2024-01-28", "predicted_sales": 1020027961.86 }
  ]
}
```

State names are case-insensitive. `california`, `California`, and `CALIFORNIA` all work.



## `GET /models`
Returns model performance scores and the winning model for every state.

```json
{
  "California": {
    "best_model": "Prophet",
    "smape_scores": {
      "XGBoost_SMAPE": 32.77,
      "Prophet_SMAPE": 20.34,
      "ARIMA_SMAPE": 40.10,
      "LSTM_SMAPE": 44.98
    }
  }
}
```

## `GET /states`
Lists all 43 valid state names.

## `GET /health`
Returns system status and number of states loaded.

```json
{ "status": "ok", "states_loaded": 43 }
```



## Models

Four models were trained and compared using SMAPE (Symmetric Mean Absolute Percentage Error) on a held out validation set consisting of the last 8 weeks of data per state.

## XGBoost
A single model trained across all 43 states simultaneously. Uses lag features, rolling statistics, and time-based features. Forecasting is done recursively, each future step is predicted using the output of the previous step.

## Prophet(Local : one per state)
Trained on original scale rather than log scale. Prophet's internal trend and seasonality decomposition performs better without the log transform. US public holidays added via `add_country_holidays`. Seasonality mode set to multiplicative, as holiday spikes grow proportionally with the overall trend.

## ARIMA(Local : one per state)
Implemented using `auto_arima` with `m=52` for weekly yearly seasonality. Consistently underperformed relative to XGBoost and Prophet across all states.

## LSTM
Single model with a state embedding layer trained across all states. Limited by the small dataset size (~226 points per state). Underperformed on the validation set and did not win any states in the final tournament.



## Model Performance Summary

| Metric | XGBoost | Prophet | ARIMA | LSTM |
|--------|---------|---------|-------|------|
| States won | 31 | 12 | 0 | 0 |
| Avg SMAPE | ~16% | ~19% | ~36% | ~30% |
| Best state | Nebraska (10.98%) | Texas (18.51%) | — | — |

XGBoost dominates stable mid size states where lag features capture the pattern well. Prophet wins large states : California, Texas, Florida, where trend decomposition handles the scale better than recursive lag-based prediction.

---

## Feature Engineering

All features were computed per state on log-transformed sales (`np.log1p`), except for Prophet and ARIMA which uses original scale.

| Feature | Description |
|---------|-------------|
| `lag_1` | Sales 1 week ago |
| `lag_7` | Sales 7 weeks ago |
| `lag_30` | Sales 30 weeks ago |
| `rolling_mean_4` | 4-week rolling average (shift-1 to prevent leakage) |
| `rolling_std_4` | 4-week rolling std (shift-1 to prevent leakage) |
| `month` | Calendar month |
| `week_of_year` | ISO week number |
| `is_holiday_week` | 1 if any US federal holiday falls within the week |

Day of week was excluded after resampling to weekly frequency cause it becomes a constant column and carries no information.

