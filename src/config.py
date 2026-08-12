"""Shared paths and constants used across the pipeline, models, and notebooks."""

from pathlib import Path

# project root is one level up from this file
ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

JOINED_DATA_PATH = PROCESSED_DATA_DIR / "load_weather_joined.parquet"
WEATHER_MODEL_PREDICTIONS_PATH = (
    PROCESSED_DATA_DIR / "weather_model_test_predictions.parquet"
)
NO_WEATHER_MODEL_PREDICTIONS_PATH = (
    PROCESSED_DATA_DIR / "no_weather_model_test_predictions.parquet"
)
BASELINE_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "baseline_test_predictions.parquet"
COMPARISON_TABLE_PATH = OUTPUTS_DIR / "comparison_table.csv"
LLM_EXPLANATIONS_PATH = OUTPUTS_DIR / "llm_explanations.csv"

# Houston coordinates, used as the weather proxy point for the Coast zone
HOUSTON_LAT = 29.7604
HOUSTON_LON = -95.3698
HOUSTON_ELEVATION_M = 12

# column name gridstatus uses for the Coast weather zone load (confirmed from
# gridstatus 0.36.0 source, _process_post_settlements_load_data renames it to this)
ERCOT_WEATHER_ZONE = "Coast"

LOCAL_TIMEZONE = "America/Chicago"

# how many full years to hold out as the test set (most recent N years)
TEST_SET_YEARS = 1

# extreme day = daily min or max temp in the top/bottom X percent, by training data only
EXTREME_TEMP_PERCENTILE = 5

# year range to try pulling ERCOT load for, missing years are skipped automatically
LOAD_START_YEAR = 2018
LOAD_END_YEAR = 2025

GROQ_MODEL = "openai/gpt-oss-120b"

RANDOM_SEED = 42
