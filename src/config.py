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

# per-row SHAP contributions for the weather-aware model's top features, saved
# alongside its predictions so the insight layer can say what actually drove
# a forecast without re-running SHAP every time
SHAP_CONTRIBUTIONS_PATH = PROCESSED_DATA_DIR / "weather_model_test_shap.parquet"

# cache of already-generated LLM insights, keyed by date range, so revisiting
# a day/week/month/year in the app doesn't spend another API call
LLM_INSIGHT_CACHE_PATH = OUTPUTS_DIR / "llm_insight_cache.json"

# Houston coordinates, used as the weather proxy point for the Coast zone
HOUSTON_LAT = 29.7604
HOUSTON_LON = -95.3698
HOUSTON_ELEVATION_M = 12

# column name gridstatus uses for the Coast weather zone load
ERCOT_WEATHER_ZONE = "Coast"

LOCAL_TIMEZONE = "America/Chicago"

# how many full years to hold out as the test set (most recent N years)
TEST_SET_YEARS = 1

# extreme day = daily min or max temp in the top/bottom X percent, by training data only
EXTREME_TEMP_PERCENTILE = 5

# years of ERCOT archive data to use. 2026 is skipped, its archive file is
# corrupted and only covers a partial year anyway. 2019-2025 gives 7 full
# years, including Winter Storm Uri (Feb 2021) and the 2023 summer heat records.
LOAD_START_YEAR = 2019
LOAD_END_YEAR = 2025

# where downloaded ERCOT archive files live, see download_archive.py
ARCHIVE_DIR = RAW_DATA_DIR / "manual_archive"

# direct links to ERCOT's per-year load archive files, from their
# "Hourly Load Data Archives" page (ercot.com/gridinfo/load/load_hist)
ERCOT_ARCHIVE_URLS = {
    # kept for reference only, not fetched -- see LOAD_END_YEAR above
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
