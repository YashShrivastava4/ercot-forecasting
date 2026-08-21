"""Baseline forecasting model: SARIMAX using only load history and calendar
features (hour, day of week, time of year, holiday flag). No temperature."""

import pickle

import holidays
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src import config, utils

DEFAULT_ORDER = (2, 1, 2)


def build_calendar_features(index: pd.DatetimeIndex, fourier_harmonics: int = 3) -> pd.DataFrame:
    """Calendar-only exogenous regressors, no load history and no weather.
    Fourier terms stand in for hour/day-of-year seasonality so it stays fast to fit.
    """
    hour_of_day = index.hour + index.minute / 60
    day_of_year = index.dayofyear

    features = {}
    for k in range(1, fourier_harmonics + 1):
        features[f"hour_sin_{k}"] = np.sin(2 * np.pi * k * hour_of_day / 24)
        features[f"hour_cos_{k}"] = np.cos(2 * np.pi * k * hour_of_day / 24)
        features[f"doy_sin_{k}"] = np.sin(2 * np.pi * k * day_of_year / 365.25)
        features[f"doy_cos_{k}"] = np.cos(2 * np.pi * k * day_of_year / 365.25)

    exog = pd.DataFrame(features, index=index)

    # fixed category list so train/test always produce the same dummy columns,
    # even if one split happens to be missing a weekday (e.g. a short test set)
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_of_week = pd.Categorical(index.day_name(), categories=day_names)
    # Monday is the dropped reference level
    dow_dummies = pd.get_dummies(day_of_week, prefix="dow", drop_first=True)
    dow_dummies.index = index
    exog = exog.join(dow_dummies)

    tx_holidays = holidays.US(state="TX", years=range(index.year.min(), index.year.max() + 1))
    holiday_dates = set(tx_holidays.keys())
    exog["is_holiday"] = pd.Series(index.date, index=index).isin(holiday_dates).astype(int)

    return exog.astype(float)


def fit_baseline_model(train_load: pd.Series, train_exog: pd.DataFrame, order=DEFAULT_ORDER):
    """Fit the calendar-only SARIMAX model on training data."""
    model = SARIMAX(
        train_load,
        exog=train_exog,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    print("fitting baseline SARIMAX, this can take a few minutes on multi-year hourly data")
    return model.fit(disp=False)


def forecast_baseline(fitted_model, test_exog: pd.DataFrame) -> pd.Series:
    """One-shot forecast over the full test horizon, not walk-forward."""
    forecast = fitted_model.get_forecast(steps=len(test_exog), exog=test_exog)
    predicted = forecast.predicted_mean
    predicted.index = test_exog.index
    return predicted.rename("load_mw_pred")


def save_model(fitted_model, path=config.MODELS_DIR / "baseline_sarimax.pkl") -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(fitted_model, f)
    print(f"saved baseline model to {path}")


def save_test_predictions(actual: pd.Series, predicted: pd.Series) -> None:
    """Save actual vs predicted so later steps can reuse it without refitting."""
    out = pd.DataFrame({"actual_mw": actual, "predicted_mw": predicted})
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(config.BASELINE_PREDICTIONS_PATH)
    print(f"saved test predictions to {config.BASELINE_PREDICTIONS_PATH}")


def run_baseline() -> tuple[pd.Series, pd.Series]:
    """End to end: load processed data, split, fit, forecast, report overall error."""
    df = pd.read_parquet(config.JOINED_DATA_PATH).asfreq("h")
    train, test = utils.time_ordered_split(df)

    train_exog = build_calendar_features(train.index)
    test_exog = build_calendar_features(test.index)

    fitted = fit_baseline_model(train["load_mw"], train_exog)
    save_model(fitted)

    predicted = forecast_baseline(fitted, test_exog)
    actual = test["load_mw"]

    print(f"baseline overall MAPE: {utils.mape(actual, predicted):.2f}%")
    print(f"baseline overall RMSE: {utils.rmse(actual, predicted):.1f} MW")

    save_test_predictions(actual, predicted)
    return actual, predicted


if __name__ == "__main__":
    run_baseline()
