"""Weather-aware forecasting model: XGBoost using load history, calendar
features, and temperature (raw + deviation from the historical norm)."""

import pickle

import holidays
import pandas as pd
import shap
from xgboost import XGBRegressor

from src import config, utils

LAG_HOURS = [1, 2, 3, 24, 48, 168]


def build_lag_and_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag/rolling/calendar features, built on the full series before any split
    so lag values at the start of the test set can still see into training data."""
    feat = pd.DataFrame(index=df.index)
    feat["load_mw"] = df["load_mw"]
    feat["temp_c"] = df["temp_c"]

    for lag in LAG_HOURS:
        feat[f"load_lag_{lag}h"] = df["load_mw"].shift(lag)

    # shift(1) first so today's own value never leaks into its own rolling average
    feat["load_roll_mean_24h"] = df["load_mw"].shift(1).rolling(24).mean()
    feat["load_roll_mean_168h"] = df["load_mw"].shift(1).rolling(168).mean()

    feat["hour"] = df.index.hour
    feat["day_of_week"] = df.index.dayofweek
    feat["month"] = df.index.month
    feat["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    tx_holidays = holidays.US(
        state="TX", years=range(df.index.year.min(), df.index.year.max() + 1)
    )
    holiday_dates = set(tx_holidays.keys())
    feat["is_holiday"] = (
        pd.Series(df.index.date, index=df.index).isin(holiday_dates).astype(int)
    )

    return feat


def compute_day_of_year_climatology(train_df: pd.DataFrame) -> pd.Series:
    """Average temp per day-of-year, from training data only, reused as-is on test data."""
    climatology = train_df.groupby(train_df.index.dayofyear)["temp_c"].mean()
    climatology = climatology.reindex(range(1, 367))
    return climatology.ffill().bfill()


def add_temp_deviation(df: pd.DataFrame, climatology: pd.Series) -> pd.DataFrame:
    """How far today's temp is from the historical norm for this day of the year."""
    df = df.copy()
    avg_for_day = pd.Series(df.index.dayofyear, index=df.index).map(climatology)
    df["temp_deviation"] = df["temp_c"] - avg_for_day
    return df


def fit_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series, valid_fraction: float = 0.1
) -> XGBRegressor:
    """Fit with early stopping on a time-ordered tail slice of training data."""
    split_idx = int(len(X_train) * (1 - valid_fraction))
    X_fit, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_fit, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=config.RANDOM_SEED,
        early_stopping_rounds=50,
        eval_metric="mae",
    )
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    print(f"stopped at iteration {model.best_iteration} of 1000")
    return model


def run_shap_importance(
    model: XGBRegressor, X_test: pd.DataFrame, top_n: int = 10
) -> tuple[pd.Series, "shap.Explanation"]:
    """Mean absolute SHAP value per feature, confirms temperature actually drives the model.
    Also returns the raw per-row values so they can be saved for the insight layer."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    importance = pd.Series(
        abs(shap_values.values).mean(axis=0), index=X_test.columns
    ).sort_values(ascending=False)

    print(f"top {top_n} features by mean |SHAP value|:")
    print(importance.head(top_n))
    return importance, shap_values


def save_shap_contributions(
    shap_values: "shap.Explanation",
    X_test: pd.DataFrame,
    importance: pd.Series,
    top_n: int = 10,
    path=None,
) -> None:
    """Save each hour's SHAP contribution (in MW) for the top features, so later
    insight generation can say what actually pushed a forecast up or down instead
    of guessing -- no need to re-run the model or SHAP just to explain a day."""
    path = path or config.SHAP_CONTRIBUTIONS_PATH
    top_features = importance.head(top_n).index.tolist()
    contributions = pd.DataFrame(
        shap_values.values, index=X_test.index, columns=X_test.columns
    )[top_features]
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    contributions.to_parquet(path)
    print(f"saved per-row SHAP contributions for top {top_n} features to {path}")


def save_model(model: XGBRegressor, path=None) -> None:
    path = path or (config.MODELS_DIR / "weather_xgboost.json")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(path)
    print(f"saved model to {path}")


def save_test_predictions(
    y_test: pd.Series, predicted: pd.Series, temp_deviation: pd.Series, path=None
) -> None:
    """Save actual vs predicted plus temp deviation, other steps (extreme-day analysis,
    LLM insights) read this instead of re-running the full model."""
    path = path or config.WEATHER_MODEL_PREDICTIONS_PATH
    out = pd.DataFrame(
        {
            "actual_mw": y_test,
            "predicted_mw": predicted,
            "temp_deviation": temp_deviation,
        }
    )
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    print(f"saved test predictions to {path}")


def run_weather_model(
    use_weather: bool = True,
) -> tuple[pd.Series, pd.Series, XGBRegressor, pd.DataFrame]:
    """End to end: build features, split, fit, forecast, report error, run SHAP.
    use_weather=False fits the same model with temperature removed -- the
    ablation used to isolate what weather specifically contributes.
    """
    df = pd.read_parquet(config.JOINED_DATA_PATH).asfreq("h")
    feat = build_lag_and_calendar_features(df)
    feat = (
        feat.dropna()
    )  # drops the first week where the 168h lag isn't available yet, plus any leftover gap rows

    train, test = utils.time_ordered_split(feat)

    climatology = compute_day_of_year_climatology(train)
    train = add_temp_deviation(train, climatology)
    test = add_temp_deviation(test, climatology)

    feature_cols = [c for c in train.columns if c != "load_mw"]
    if not use_weather:
        feature_cols = [
            c for c in feature_cols if c not in ("temp_c", "temp_deviation")
        ]

    X_train, y_train = train[feature_cols], train["load_mw"]
    X_test, y_test = test[feature_cols], test["load_mw"]

    model = fit_xgboost(X_train, y_train)

    label = "weather-aware" if use_weather else "no-weather ablation"
    model_path = config.MODELS_DIR / (
        "weather_xgboost.json" if use_weather else "no_weather_xgboost.json"
    )
    preds_path = (
        config.WEATHER_MODEL_PREDICTIONS_PATH
        if use_weather
        else config.NO_WEATHER_MODEL_PREDICTIONS_PATH
    )
    save_model(model, path=model_path)

    predicted = pd.Series(
        model.predict(X_test), index=X_test.index, name="load_mw_pred"
    )

    print(f"{label} overall MAPE: {utils.mape(y_test, predicted):.2f}%")
    print(f"{label} overall RMSE: {utils.rmse(y_test, predicted):.1f} MW")

    if use_weather:
        importance, shap_values = run_shap_importance(model, X_test)
        save_shap_contributions(shap_values, X_test, importance)

    save_test_predictions(y_test, predicted, test["temp_deviation"], path=preds_path)

    return y_test, predicted, model, X_test


if __name__ == "__main__":
    run_weather_model(use_weather=True)
    run_weather_model(use_weather=False)
