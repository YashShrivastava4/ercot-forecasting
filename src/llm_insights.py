"""For the days the weather-aware model got most wrong, generate a short plain
English explanation. The LLM only phrases numbers we've already calculated,
it never does the arithmetic itself."""

import os

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from src import config

load_dotenv()

TOP_N_DEFAULT = 10


def daily_error_summary(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse hourly actual/predicted/temp_deviation down to one row per day."""
    hourly_abs_error = (predictions_df["actual_mw"] - predictions_df["predicted_mw"]).abs()

    daily = pd.DataFrame({
        "actual_mw_avg": predictions_df["actual_mw"].groupby(predictions_df.index.date).mean(),
        "predicted_mw_avg": predictions_df["predicted_mw"].groupby(predictions_df.index.date).mean(),
        "mean_abs_error_mw": hourly_abs_error.groupby(predictions_df.index.date).mean(),
        "temp_deviation_avg": predictions_df["temp_deviation"].groupby(predictions_df.index.date).mean(),
    })
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"
    return daily


def top_error_days(daily_summary: pd.DataFrame, top_n: int = TOP_N_DEFAULT) -> pd.DataFrame:
    return daily_summary.sort_values("mean_abs_error_mw", ascending=False).head(top_n)


def build_prompt(date: pd.Timestamp, row: pd.Series) -> str:
    """Hand the model already-calculated numbers, its only job is turning them into a sentence."""
    direction = "above" if row["temp_deviation_avg"] > 0 else "below"
    return (
        f"On {date.strftime('%B %d, %Y')}, actual average hourly electricity demand was "
        f"{row['actual_mw_avg']:.0f} MW, the model predicted {row['predicted_mw_avg']:.0f} MW, "
        f"an average error of {row['mean_abs_error_mw']:.0f} MW. That day's temperature was "
        f"{abs(row['temp_deviation_avg']):.1f}°C {direction} the historical norm for that day of year. "
        "Write one plain-English sentence explaining this forecast miss for a non-technical reader. "
        "Use only the numbers given above, don't invent or estimate anything else."
    )


def generate_explanation(client: Groq, prompt: str) -> str:
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()


def generate_all_explanations(top_days: pd.DataFrame, dry_run: bool = False) -> pd.DataFrame:
    """Set dry_run=True to check the prompts/output shape without spending API calls."""
    client = None if dry_run else Groq(api_key=os.environ["GROQ_API_KEY"])

    explanations = []
    for date, row in top_days.iterrows():
        prompt = build_prompt(date, row)
        if dry_run:
            explanations.append(f"[dry run, prompt was]: {prompt}")
        else:
            explanations.append(generate_explanation(client, prompt))
        print(f"explained {date.date()}")

    result = top_days.copy()
    result["explanation"] = explanations
    return result


def run_llm_insights(top_n: int = TOP_N_DEFAULT, dry_run: bool = False) -> pd.DataFrame:
    predictions_df = pd.read_parquet(config.WEATHER_MODEL_PREDICTIONS_PATH)
    daily_summary = daily_error_summary(predictions_df)
    top_days = top_error_days(daily_summary, top_n)

    result = generate_all_explanations(top_days, dry_run=dry_run)

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(config.LLM_EXPLANATIONS_PATH)
    print(f"saved {len(result)} explanations to {config.LLM_EXPLANATIONS_PATH}")
    return result


if __name__ == "__main__":
    run_llm_insights()
