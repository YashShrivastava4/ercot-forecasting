"""Pulls ERCOT Coast-zone hourly load and Houston hourly weather, joins them on
timestamp, and saves one clean parquet file for the rest of the project to use.

This is a one-time, local step. Run it once, then commit the parquet it
produces (data/processed/load_weather_joined.parquet) to the repo. Nothing
downstream -- the notebooks, the models, or the deployed Streamlit app --
ever needs to reach ERCOT or meteostat again after that.

ERCOT load comes from files in data/raw/manual_archive/, sourced from
ERCOT's own "Hourly Load Data Archives" page
(https://www.ercot.com/gridinfo/load/load_hist). No account, no credentials,
no live scraping of that page's link list -- see src/download_archive.py.
"""

import zipfile

import gridstatus
import meteostat
import pandas as pd

from src import config
from src.download_archive import ensure_archive_files_present


def load_manual_archive_files() -> pd.DataFrame:
    """Read every ERCOT historical load file sitting in data/raw/manual_archive/.

    Handles both the newer .zip-wrapped files (2016 onward) and the older
    bare .xls files (2015 and earlier), whichever ERCOT happened to publish
    that year. gridstatus's own parser is reused here purely as a local file
    parser -- it does the DST/hour-ending/"24:00" handling ERCOT's format
    needs, and reusing it means this path is tested against the exact same
    logic gridstatus itself validates, without this project making any
    network call of its own.
    """
    if not config.ARCHIVE_DIR.exists():
        raise FileNotFoundError(
            f"{config.ARCHIVE_DIR} doesn't exist -- run "
            "`python3 -m src.download_archive` first, or see README for manual download steps"
        )

    files = sorted(
        list(config.ARCHIVE_DIR.glob("*.zip"))
        + list(config.ARCHIVE_DIR.glob("*.xls"))
        + list(config.ARCHIVE_DIR.glob("*.xlsx"))
    )
    if not files:
        raise FileNotFoundError(
            f"no archive files found in {config.ARCHIVE_DIR} -- run "
            "`python3 -m src.download_archive` first, or see README for manual download steps"
        )

    ercot = gridstatus.Ercot()
    frames = []
    # every one of these needs to exist in some form before calling the parser
    # below, since it unconditionally selects all of them at the end -- a
    # missing one (older file format, unexpected column name) shouldn't crash
    # the whole pipeline, only Coast is actually used downstream anyway
    zone_aliases = {
        "COAST": ["COAST"],
        "EAST": ["EAST"],
        "FAR_WEST": ["FWEST", "FAR_WEST"],
        "NORTH": ["NORTH"],
        "NORTH_C": ["NCENT", "NORTH_C"],
        "SOUTH": ["SOUTH", "SOUTHERN"],
        "SOUTH_C": ["SCENT", "SOUTH_C"],
        "WEST": ["WEST"],
        "ERCOT": ["ERCOT", "TOTAL"],
    }

    for path in files:
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as zf:
                inner_name = zf.namelist()[0]
                raw = pd.read_excel(zf.open(inner_name))
        else:
            raw = pd.read_excel(path)

        raw.columns = raw.columns.str.strip()
        for canonical, aliases in zone_aliases.items():
            if not any(a in raw.columns for a in aliases):
                raw[canonical] = pd.NA

        parsed = ercot._process_post_settlements_load_data(raw)
        print(f"loaded {len(parsed)} rows from {path.name}")
        frames.append(parsed)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="Interval Start").sort_values(
        "Interval Start"
    )
    combined = combined.reset_index(drop=True)

    print(
        f"combined ERCOT load: {combined['Interval Start'].min()} to "
        f"{combined['Interval Start'].max()}, {len(combined)} rows"
    )

    # a real gap in the archive (a year you haven't downloaded yet) should be
    # visible, not silently modeled across
    gap_hours = combined["Interval Start"].diff().dt.total_seconds().div(3600)
    big_gaps = gap_hours[gap_hours > 24]
    if not big_gaps.empty:
        print(
            f"note: {len(big_gaps)} gap(s) longer than a day in the combined load series "
            f"(largest: {big_gaps.max():.0f}h) -- check whether this is a year you're "
            f"missing an archive file for in {config.ARCHIVE_DIR}"
        )

    return combined


def fetch_ercot_load() -> pd.DataFrame:
    """Make sure the known archive years are downloaded, then parse everything present."""
    ensure_archive_files_present()
    return load_manual_archive_files()


def _station_has_hourly_data(station_id: str, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """True if this station returns any hourly rows for the given window."""
    ts = meteostat.hourly(station_id, start, end, timezone=config.LOCAL_TIMEZONE)
    df = ts.fetch()
    return df is not None and not df.empty


def select_houston_station(start: pd.Timestamp, end: pd.Timestamp, probe_days: int = 7) -> str:
    """Pick the closest station to the Houston point that actually has hourly
    data across the whole requested range, not just the closest one on paper.

    Nearest-by-distance isn't the same as having data -- a small local
    station can be geographically closest but have a sparse or empty
    archive. This walks candidates in distance order and probes a short
    window near the start of each calendar year in range (a full multi-year
    probe would hit meteostat's 3-year request cap), picking the first
    station that has data for every one of those years.
    """
    point = meteostat.Point(
        config.HOUSTON_LAT, config.HOUSTON_LON, config.HOUSTON_ELEVATION_M
    )
    candidates = meteostat.stations.nearby(point)
    probe_years = range(start.year, end.year + 1)

    for station_id, row in candidates.iterrows():
        print(
            f"checking station {station_id} ({row['name']}, {row['distance'] / 1000:.1f} km away)"
        )
        covers_full_range = True
        for year in probe_years:
            probe_start = max(start, pd.Timestamp(year, 1, 1))
            probe_end = min(end, probe_start + pd.Timedelta(days=probe_days))
            if not _station_has_hourly_data(station_id, probe_start, probe_end):
                print(f"  no data around {probe_start.date()}, skipping this station")
                covers_full_range = False
                break
        if covers_full_range:
            print(f"selected station {station_id} ({row['name']}) -- has data across the full range")
            return station_id

    raise RuntimeError(
        f"no candidate station near the Houston point has hourly data covering "
        f"{start.date()} to {end.date()} -- checked {len(candidates)} candidates"
    )


def fetch_houston_weather(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Pull hourly weather for a station near the Houston point that actually has data for this range, already localized to Central time."""
    station_id = select_houston_station(start, end)

    # meteostat blocks hourly requests spanning more than 3 years by default,
    # so pull one calendar year at a time and stitch the results together
    yearly_frames = []
    for year in range(start.year, end.year + 1):
        year_start = max(start, pd.Timestamp(year, 1, 1))
        year_end = min(end, pd.Timestamp(year, 12, 31, 23, 59))
        print(f"pulling Houston weather for {year}")
        ts = meteostat.hourly(station_id, year_start, year_end, timezone=config.LOCAL_TIMEZONE)
        year_df = ts.fetch()
        if year_df is None or year_df.empty:
            print(f"  no weather data returned for {year}")
            continue
        yearly_frames.append(year_df)

    if not yearly_frames:
        raise RuntimeError("meteostat returned no weather data for this range")

    return pd.concat(yearly_frames).sort_index()


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
    raw_load = fetch_ercot_load()
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
