"""Small shared helpers: error metrics and extreme-day flagging, kept in one
place so the threshold logic stays identical everywhere it's used."""

import numpy as np
import pandas as pd

from src import config


def _drop_unpaired_nans(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Drop any position where either side is NaN, e.g. an unresolved data gap."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    dropped = len(y_true) - mask.sum()
    if dropped > 0:
        print(
            f"  ({dropped} rows skipped in this metric due to NaN actual/predicted values)"
        )
    return y_true[mask], y_pred[mask]


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean absolute percentage error, as a plain percentage."""
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    y_true, y_pred = _drop_unpaired_nans(y_true, y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    y_true, y_pred = _drop_unpaired_nans(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def daily_min_max_temp(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse hourly temp readings down to one min/max row per calendar day."""
    daily = hourly_df["temp_c"].groupby(hourly_df.index.date).agg(["min", "max"])
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"
    return daily.rename(columns={"min": "temp_min", "max": "temp_max"})


def compute_extreme_thresholds(train_daily_temp: pd.DataFrame) -> dict:
    """Compute the extreme-day percentile cutoffs from training data only.
    Reused as a fixed cutoff on the test set -- never recomputed with test data.
    """
    pct = config.EXTREME_TEMP_PERCENTILE
    return {
        "low_cutoff": float(np.nanpercentile(train_daily_temp["temp_min"], pct)),
        "high_cutoff": float(np.nanpercentile(train_daily_temp["temp_max"], 100 - pct)),
    }


def flag_extreme_days(daily_temp: pd.DataFrame, thresholds: dict) -> pd.Series:
    """Mark a day extreme if its min or max temp crosses the fixed training-set cutoffs."""
    is_extreme = (daily_temp["temp_min"] <= thresholds["low_cutoff"]) | (
        daily_temp["temp_max"] >= thresholds["high_cutoff"]
    )
    return is_extreme.rename("is_extreme_day")


def segment_metrics_by_extreme_day(
    predictions_df: pd.DataFrame, extreme_dates: set
) -> dict:
    """Split hourly actual/predicted rows into normal vs extreme days and score each group.
    predictions_df needs actual_mw/predicted_mw columns; extreme_dates comes from flag_extreme_days().
    """
    is_extreme = pd.Series(predictions_df.index.date, index=predictions_df.index).isin(
        extreme_dates
    )
    normal = predictions_df[~is_extreme.values]
    extreme = predictions_df[is_extreme.values]

    # count days actually present in this predictions data, not the full theoretical
    # extreme-day set — if a day is missing here (e.g. dropped upstream) it shouldn't be counted
    present_dates = set(predictions_df.index.date)

    return {
        "normal_mape": mape(normal["actual_mw"], normal["predicted_mw"]),
        "normal_rmse": rmse(normal["actual_mw"], normal["predicted_mw"]),
        "extreme_mape": mape(extreme["actual_mw"], extreme["predicted_mw"]),
        "extreme_rmse": rmse(extreme["actual_mw"], extreme["predicted_mw"]),
        "extreme_day_count": len(present_dates & extreme_dates),
        "normal_day_count": len(present_dates - extreme_dates),
    }


def time_ordered_split(df: pd.DataFrame, test_years: int = config.TEST_SET_YEARS):
    """Hold out the most recent N full calendar years as the test set, never shuffle."""
    last_year = df.index.max().year
    year_end = pd.Timestamp(f"{last_year}-12-31 23:00", tz=df.index.tz)
    if df.index.max() < year_end:
        last_year -= 1  # last year in the data is partial, don't count it as held-out

    test_start_year = last_year - test_years + 1
    test_start = pd.Timestamp(f"{test_start_year}-01-01", tz=df.index.tz)

    train = df[df.index < test_start]
    test = df[(df.index >= test_start) & (df.index.year <= last_year)]
    print(
        f"train: {train.index.min().date()} to {train.index.max().date()} ({len(train)} rows)"
    )
    print(
        f"test:  {test.index.min().date()} to {test.index.max().date()} ({len(test)} rows)"
    )
    return train, test
