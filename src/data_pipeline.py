"""Pulls ERCOT Coast-zone hourly load and Houston hourly weather, joins them on
timestamp, and saves one clean parquet file for the rest of the project to use."""

import gridstatus
import meteostat
import pandas as pd

from src import config


def fetch_ercot_load(start_year: int, end_year: int) -> pd.DataFrame:
    """Pull hourly Coast-zone load year by year, skipping any year ERCOT hasn't published."""
    ercot = gridstatus.Ercot()
    yearly_frames = []

    for year in range(start_year, end_year + 1):
        try:
            print(f"pulling ERCOT load for {year}")
            year_df = ercot.get_hourly_load_post_settlements(date=str(year))
            yearly_frames.append(year_df)
        except Exception as e:
            # ERCOT hasn't published this year yet, or the request failed, either way skip it
            print(f"  skipping {year}: {e}")

    if not yearly_frames:
        raise RuntimeError("no ERCOT load data could be pulled for any year in range")

    load_df = pd.concat(yearly_frames, ignore_index=True)
    load_df = load_df.drop_duplicates(subset="Interval Start").sort_values(
        "Interval Start"
    )
    return load_df.reset_index(drop=True)


def fetch_houston_weather(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Pull hourly weather for the Houston point, already localized to Central time."""
    point = meteostat.Point(
        config.HOUSTON_LAT, config.HOUSTON_LON, config.HOUSTON_ELEVATION_M
    )
    print(f"pulling Houston weather from {start.date()} to {end.date()}")
    ts = meteostat.hourly(point, start, end, timezone=config.LOCAL_TIMEZONE)
    weather_df = ts.fetch()

    if weather_df is None or weather_df.empty:
        raise RuntimeError("meteostat returned no weather data for this range")

    return weather_df


def prepare_load_series(load_df: pd.DataFrame) -> pd.DataFrame:
    """Keep just the Coast zone load, indexed by the hour it covers."""
    if config.ERCOT_WEATHER_ZONE not in load_df.columns:
        raise KeyError(
            f"'{config.ERCOT_WEATHER_ZONE}' column not found, actual columns are: "
            f"{load_df.columns.tolist()}"
        )

    load_series = load_df.set_index("Interval Start")[config.ERCOT_WEATHER_ZONE]
    load_series = load_series.rename("load_mw")
    load_series.index = load_series.index.tz_convert(config.LOCAL_TIMEZONE)
    load_series.index.name = "timestamp"

    # a handful of NaN rows show up in raw ERCOT files, drop rather than guess a fill
    return load_series.dropna().to_frame()


def prepare_weather_series(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Pull out the columns we actually need and rename temp for clarity."""
    weather_df = weather_df.copy()
    weather_df.index.name = "timestamp"

    if weather_df.index.tz is None:
        weather_df.index = weather_df.index.tz_localize(config.LOCAL_TIMEZONE)
    else:
        weather_df.index = weather_df.index.tz_convert(config.LOCAL_TIMEZONE)

    keep_cols = [c for c in ["temp", "dwpt", "rhum"] if c in weather_df.columns]
    weather_df = weather_df[keep_cols].rename(columns={"temp": "temp_c"})
    return weather_df


def check_dst_days(df: pd.DataFrame) -> None:
    """Print any day that doesn't have exactly 24 hourly rows, so DST issues are visible not silent."""
    hours_per_day = df.groupby(df.index.date).size()
    odd_days = hours_per_day[hours_per_day != 24]

    if odd_days.empty:
        print("every day in the joined data has exactly 24 hourly rows")
        return

    print(
        f"found {len(odd_days)} day(s) without exactly 24 rows (expected on DST changeover days):"
    )
    print(odd_days)


def close_hourly_gaps(df: pd.DataFrame, max_interpolate_hours: int = 3) -> pd.DataFrame:
    """Reindex onto a complete, regularly-spaced hourly index.

    Both the ERCOT and meteostat sides can be missing individual hours (a
    dropped NaN row, a gap in the weather station's record), and an inner
    join just silently skips those timestamps. If that's left alone, every
    shift()/rolling() feature built later on treats "N rows back" as "N hours
    back", which is only true if the index has zero gaps. Short gaps get
    interpolated, longer ones are left as NaN and reported rather than guessed.
    """
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h", tz=df.index.tz)
    missing_before = full_index.difference(df.index)

    df = df.reindex(full_index)
    df.index.name = "timestamp"

    if len(missing_before) == 0:
        print("no missing hours, index is already a complete hourly grid")
        return df

    print(
        f"{len(missing_before)} hour(s) missing from the joined data, reindexed onto a complete hourly grid"
    )
    df = df.interpolate(method="time", limit=max_interpolate_hours, limit_area="inside")

    still_missing = df[df["load_mw"].isna() | df["temp_c"].isna()]
    if not still_missing.empty:
        print(
            f"{len(still_missing)} hour(s) still missing after interpolating gaps up to {max_interpolate_hours}h "
            f"(gap too long, or at the very start/end) — left as NaN, downstream steps drop or ignore these:"
        )
        print(still_missing.index.to_series().groupby(still_missing.index.date).count())

    return df


def join_load_and_weather(
    load_series: pd.DataFrame, weather_series: pd.DataFrame
) -> pd.DataFrame:
    """Join on the shared hourly timestamp index, close any gaps, and report how much data survived."""
    joined = load_series.join(weather_series, how="inner")
    print(
        f"load rows: {len(load_series)}, weather rows: {len(weather_series)}, joined rows: {len(joined)}"
    )

    dropped = len(load_series) - len(joined)
    if dropped > 0:
        print(f"{dropped} load hours had no matching weather reading and were dropped")

    joined = close_hourly_gaps(joined)
    check_dst_days(joined)
    return joined


def save_processed(df: pd.DataFrame) -> None:
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.JOINED_DATA_PATH)
    print(f"saved {len(df)} rows to {config.JOINED_DATA_PATH}")


def run_pipeline() -> pd.DataFrame:
    raw_load = fetch_ercot_load(config.LOAD_START_YEAR, config.LOAD_END_YEAR)
    load_series = prepare_load_series(raw_load)

    start = load_series.index.min().tz_localize(None)
    end = load_series.index.max().tz_localize(None)
    raw_weather = fetch_houston_weather(start, end)
    weather_series = prepare_weather_series(raw_weather)

    joined = join_load_and_weather(load_series, weather_series)
    save_processed(joined)
    return joined


if __name__ == "__main__":
    run_pipeline()
