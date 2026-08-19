"""Shared paths and constants used across the pipeline, models, and notebooks."""

from pathlib import Path

# project root is one level up from this file
ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

JOINED_DATA_PATH = PROCESSED_DATA_DIR / "load_weather_joined.parquet"
WEATHER_MODEL_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "weather_model_test_predictions.parquet"
NO_WEATHER_MODEL_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "no_weather_model_test_predictions.parquet"
BASELINE_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "baseline_test_predictions.parquet"
COMPARISON_TABLE_PATH = OUTPUTS_DIR / "comparison_table.csv"
LLM_EXPLANATIONS_PATH = OUTPUTS_DIR / "llm_explanations.csv"

# Houston coordinates, used as the weather proxy point for the Coast zone
HOUSTON_LAT = 29.7604
HOUSTON_LON = -95.3698
HOUSTON_ELEVATION_M = 12

# column name gridstatus uses for the Coast weather zone load (confirmed from
# gridstatus 0.36.0 source, _process_post_settlements_load_data renames it to
# this). gridstatus is only used here as a local file parser for the archive
# files below -- it never makes a network call anywhere in this project.
ERCOT_WEATHER_ZONE = "Coast"

LOCAL_TIMEZONE = "America/Chicago"

# how many full years to hold out as the test set (most recent N years)
TEST_SET_YEARS = 1

# extreme day = daily min or max temp in the top/bottom X percent, by training data only
EXTREME_TEMP_PERCENTILE = 5

# years of ERCOT archive data to use. 2026 was tried and dropped: both the
# auto-download and a manual re-download produced a zip that fails CRC
# validation -- checked directly, the compressed stream itself is short
# versus what the zip's own central directory declares, so the download is
# truncated, not just a bad link or a "2026 isn't published yet" thing. It's
# also only a partial year regardless (we're mid-2026), so it wouldn't give
# a usable full year even with a clean download. 2019-2025 is 7 complete
# years, including Winter Storm Uri (Feb 2021) and the 2023 summer heat
# records -- both genuinely useful for the extreme-day story, not filler.
LOAD_START_YEAR = 2019
LOAD_END_YEAR = 2025

# where downloaded/placed ERCOT archive files live -- see download_archive.py
ARCHIVE_DIR = RAW_DATA_DIR / "manual_archive"

# direct download links for ERCOT's own "Hourly Load Data Archives" page
# (https://www.ercot.com/gridinfo/load/load_hist), confirmed working as of
# Aug 2026. Each year's file is hosted at a URL that encodes its publish
# date, so a URL from a past year is stable, but next year's URL can't be
# predicted in advance -- download_archive.py handles that gracefully by
# telling you to add the new one here rather than failing silently.
ERCOT_ARCHIVE_URLS = {
    # kept for reference, not fetched -- LOAD_END_YEAR stops at 2025, see the
    # note above on why 2026's archive is excluded
    2026: "https://www.ercot.com/files/docs/2026/02/10/Native_Load_2026.zip",
    2025: "https://www.ercot.com/files/docs/2025/02/11/Native_Load_2025.zip",
    2024: "https://www.ercot.com/files/docs/2024/02/06/Native_Load_2024.zip",
    2023: "https://www.ercot.com/files/docs/2023/02/09/Native_Load_2023.zip",
    2022: "https://www.ercot.com/files/docs/2022/02/08/Native_Load_2022.zip",
    2021: "https://www.ercot.com/files/docs/2021/11/12/Native_Load_2021.zip",
    2020: "https://www.ercot.com/files/docs/2021/01/12/Native_Load_2020.zip",
    2019: "https://www.ercot.com/files/docs/2020/01/09/Native_Load_2019.zip",
    2018: "https://www.ercot.com/files/docs/2019/01/07/native_load_2018.zip",
}

GROQ_MODEL = "openai/gpt-oss-120b"

RANDOM_SEED = 42
