"""On-demand insight layer for the weather-aware model. For any day, week,
month, or the full year, builds the real numbers first in Python -- error,
temperature context, extreme-day flag, and each hour's actual SHAP drivers --
then asks the LLM to turn those into a short model-side explanation and a
short business-side one. Results are cached to disk by date range, so
re-viewing something already looked at costs no extra API call, but nothing
stops a genuinely new day or period from getting its own real explanation.
"""

import datetime as dt
import hashlib
import json
import os

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from src import config, utils

load_dotenv()

TOP_N_DEFAULT = 10
# bump this whenever build_prompt's wording changes, so cached insights from
# the old prompt (e.g. the over/under-prediction bias fix) invalidate instead
# of silently serving stale text with the same underlying facts
PROMPT_VERSION = "v2-bullets-bias-fix"


def daily_error_summary(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse hourly actual/predicted/temp_deviation down to one row per day."""
    hourly_signed_error = predictions_df["predicted_mw"] - predictions_df["actual_mw"]
    hourly_abs_error = hourly_signed_error.abs()

    daily = pd.DataFrame(
        {
            "actual_mw_avg": predictions_df["actual_mw"]
            .groupby(predictions_df.index.date)
            .mean(),
            "predicted_mw_avg": predictions_df["predicted_mw"]
            .groupby(predictions_df.index.date)
            .mean(),
            "mean_abs_error_mw": hourly_abs_error.groupby(
                predictions_df.index.date
            ).mean(),
            # signed, not absolute -- tells us whether the model runs high or low, not just by how much
            "mean_signed_error_mw": hourly_signed_error.groupby(
                predictions_df.index.date
            ).mean(),
            "temp_deviation_avg": predictions_df["temp_deviation"]
            .groupby(predictions_df.index.date)
            .mean(),
        }
    )
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"
    # error as a share of actual demand, easier to judge "is this a big miss" than a raw MW number
    daily["error_pct_avg"] = (daily["mean_abs_error_mw"] / daily["actual_mw_avg"]) * 100
    return daily


def top_error_days(
    daily_summary: pd.DataFrame, top_n: int = TOP_N_DEFAULT
) -> pd.DataFrame:
    return daily_summary.sort_values("mean_abs_error_mw", ascending=False).head(top_n)


def load_shap_contributions() -> pd.DataFrame | None:
    """Per-hour SHAP contributions for the top model features, if weather_model.py
    has been run since this feature was added. None just means older model output --
    insights still work, they just can't cite a specific driver."""
    if not config.SHAP_CONTRIBUTIONS_PATH.exists():
        return None
    return pd.read_parquet(config.SHAP_CONTRIBUTIONS_PATH)


def compute_extreme_day_context() -> tuple[dict, set]:
    """Recomputes the same leakage-safe extreme-day thresholds used in the
    extreme-day analysis notebook, so an insight can say whether a day was
    flagged extreme without needing a separate saved file for it."""
    if not config.JOINED_DATA_PATH.exists():
        return {}, set()
    df = pd.read_parquet(config.JOINED_DATA_PATH).asfreq("h")
    train, test = utils.time_ordered_split(df)
    thresholds = utils.compute_extreme_thresholds(utils.daily_min_max_temp(train))
    extreme_flags = utils.flag_extreme_days(utils.daily_min_max_temp(test), thresholds)
    extreme_dates = set(extreme_flags[extreme_flags].index.date)
    return thresholds, extreme_dates


def format_period_label(start_date: dt.date, end_date: dt.date) -> str:
    if start_date == end_date:
        return start_date.strftime("%B %d, %Y")
    span_days = (end_date - start_date).days + 1
    return f"{start_date.strftime('%b %d')}\u2013{end_date.strftime('%b %d, %Y')} ({span_days}-day window)"


def period_key(start_date: dt.date, end_date: dt.date) -> str:
    return f"{start_date.isoformat()}_{end_date.isoformat()}"


def period_facts(
    daily_summary: pd.DataFrame,
    shap_df: pd.DataFrame | None,
    extreme_dates: set,
    full_year_baseline_mae: float,
    start_date: dt.date,
    end_date: dt.date,
) -> dict:
    """Every number the prompt is allowed to use for this date range, computed
    once here in plain pandas so the LLM never has to calculate or recall anything."""
    mask = (daily_summary.index.date >= start_date) & (
        daily_summary.index.date <= end_date
    )
    window = daily_summary.loc[mask]
    dates_in_window = set(window.index.date)
    extreme_in_window = sorted(dates_in_window & extreme_dates)

    mean_abs_error = float(window["mean_abs_error_mw"].mean())

    facts = {
        "num_days": int(len(window)),
        "actual_mw_avg": float(window["actual_mw_avg"].mean()),
        "predicted_mw_avg": float(window["predicted_mw_avg"].mean()),
        "mean_abs_error_mw": mean_abs_error,
        "error_pct_avg": float(window["error_pct_avg"].mean()),
        "mean_signed_error_mw": float(window["mean_signed_error_mw"].mean()),
        "temp_deviation_avg": float(window["temp_deviation_avg"].mean()),
        "extreme_day_count": len(extreme_in_window),
        "error_vs_year_baseline_pct": (
            (mean_abs_error - full_year_baseline_mae) / full_year_baseline_mae
        )
        * 100,
    }

    if shap_df is not None:
        shap_mask = (shap_df.index.date >= start_date) & (
            shap_df.index.date <= end_date
        )
        shap_window = shap_df.loc[shap_mask]
        if not shap_window.empty:
            avg_contrib = shap_window.mean()
            avg_contrib = avg_contrib.reindex(
                avg_contrib.abs().sort_values(ascending=False).index
            )
            facts["top_drivers"] = [
                (name, float(val)) for name, val in avg_contrib.head(4).items()
            ]

    return facts


def build_prompt(period_label: str, facts: dict) -> str:
    """Hand the model already-calculated numbers and ask for two short,
    grounded sections -- what the model saw, and what it means operationally."""
    direction = "above" if facts["temp_deviation_avg"] > 0 else "below"
    # signed error is predicted minus actual -- positive means the model's
    # prediction came in higher than reality, i.e. it over-predicted
    bias_direction = (
        "over-predicted" if facts["mean_signed_error_mw"] > 0 else "under-predicted"
    )
    extreme_note = (
        f"{facts['extreme_day_count']} of the day(s) in this period were flagged as "
        "statistically extreme-temperature days"
        if facts["extreme_day_count"]
        else "none of the day(s) in this period were flagged as statistically extreme-temperature days"
    )

    drivers_note = ""
    if facts.get("top_drivers"):
        parts = [
            f"{name} ({value:+.0f} MW average contribution)"
            for name, value in facts["top_drivers"]
        ]
        drivers_note = (
            "\n- The model's largest average contributing factors this period, from SHAP: "
            + ", ".join(parts)
        )

    return (
        "You are writing a short two-part note about the weather-aware ERCOT Coast (Houston) "
        f"load forecasting model's performance for {period_label}, for a reader who isn't a data scientist "
        "and doesn't want technical jargon.\n\n"
        "Facts for this period, already calculated -- treat these as the only facts you have, "
        "never invent a number, date, or event that isn't listed here:\n"
        f"- Period: {period_label} ({facts['num_days']} day(s))\n"
        f"- Actual average hourly demand: {facts['actual_mw_avg']:.0f} MW\n"
        f"- Model's predicted average hourly demand: {facts['predicted_mw_avg']:.0f} MW\n"
        f"- Average hourly forecast error: {facts['mean_abs_error_mw']:.0f} MW "
        f"({facts['error_pct_avg']:.2f}% of actual demand)\n"
        f"- On average the model {bias_direction} demand by {abs(facts['mean_signed_error_mw']):.0f} MW per hour\n"
        f"- This period's average error is {facts['error_vs_year_baseline_pct']:+.1f}% relative to the "
        "full 2025 test-year average error\n"
        f"- Average temperature was {abs(facts['temp_deviation_avg']):.1f}\u00b0C {direction} the typical "
        f"temperature for this time of year\n"
        f"- {extreme_note}"
        f"{drivers_note}\n\n"
        "Write two short labeled sections. Each section is 3 to 4 short bullet points, one line each, "
        "not a paragraph. Start every bullet with '- '. Use plain, everyday words -- say things like "
        "'ran a bit high' instead of 'positive bias', 'colder than usual for this time of year' instead "
        "of 'negative temperature deviation', 'the biggest factor was...' instead of naming SHAP or MAPE "
        "directly. Round numbers naturally the way a person would say them out loud. Keep each bullet a "
        "single, clear sentence -- no hedging, no filler phrases like 'it is worth noting'.\n\n"
        "MODEL INSIGHT:\n"
        "Bullets covering, in this order: (1) whether this period's error was about typical, better than "
        "usual, or worse than usual for this model, using the relative-error number, (2) whether the model "
        "ran high or low and by roughly how much, (3) what actually seemed to drive the forecast this "
        "period based on the SHAP drivers and temperature context -- and if temperature wasn't a top "
        "driver, say plainly that something else (like recent demand history) mattered more, rather than "
        "forcing a weather explanation.\n\n"
        "BUSINESS INSIGHT:\n"
        "Bullets covering, in plain operational terms, what this accuracy level and over/under direction "
        "would mean for a grid operator's demand planning and reserve procurement this period -- e.g. "
        "risk of under-buying or over-buying reserve power, or that the risk was minimal because the "
        "error was small. Speak generally about forecasting reliability and risk -- do not invent specific "
        "dollar costs, company policies, or actions the numbers above don't imply.\n\n"
        "Use only the numbers and facts given. Do not invent additional events, causes, or figures."
    )


def _call_groq(
    client: Groq, prompt: str, max_tokens: int, reasoning_effort: str
) -> tuple[str, str]:
    """One Groq call, returns (text, finish_reason)."""
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=max_tokens,
        # gpt-oss-120b "thinks" before answering, and that reasoning counts
        # against max_tokens -- hidden keeps it out of the saved text
        reasoning_effort=reasoning_effort,
        reasoning_format="hidden",
    )
    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    return text, choice.finish_reason


def generate_insight(client: Groq, prompt: str) -> str:
    """Two labeled sections need real room to think and to write -- generous
    token budgets here, with an automatic retry if a call gets cut off or
    comes back missing the second section."""
    text, finish_reason = _call_groq(
        client, prompt, max_tokens=1200, reasoning_effort="medium"
    )

    if finish_reason == "length" or not text or "BUSINESS INSIGHT" not in text.upper():
        print("  response got cut off or incomplete, retrying with more room")
        text, finish_reason = _call_groq(
            client, prompt, max_tokens=2000, reasoning_effort="low"
        )

    if not text:
        return "[insight generation failed for this period -- try again]"
    return text


def split_insight_sections(text: str) -> tuple[str, str]:
    """Split the two labeled sections back apart for display. Falls back to
    showing everything as the model insight if the label didn't come back exactly."""
    model_part, _, business_part = text.partition("BUSINESS INSIGHT:")
    model_part = model_part.replace("MODEL INSIGHT:", "").strip()
    business_part = business_part.strip()
    if not business_part:
        return text.strip(), ""
    return model_part, business_part


def section_to_html(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]

    if not lines:
        return ""

    html_parts = []
    in_list = False

    for line in lines:
        if line.startswith("**") and line.endswith("**"):
            if in_list:
                html_parts.append("</ul>")
                in_list = False

            heading = line.strip("*").strip()
            html_parts.append(f"<h4>{heading}</h4>")

        elif line.startswith(("- ", "* ")):
            if not in_list:
                html_parts.append('<ul class="insight-list">')
                in_list = True

            bullet = line[2:].strip()
            html_parts.append(f"<li>{bullet}</li>")

        else:
            if in_list:
                html_parts.append("</ul>")
                in_list = False

            html_parts.append(f"<p>{line}</p>")

    if in_list:
        html_parts.append("</ul>")

    return "".join(html_parts)


class InsightCache:
    """Disk-backed cache so revisiting a day/period already seen doesn't spend
    another API call. Keyed by date range plus a fingerprint of the facts that
    went into the prompt, so re-running the models invalidates stale entries
    instead of silently serving an outdated number."""

    def __init__(self, path=None):
        self.path = path or config.LLM_INSIGHT_CACHE_PATH
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    @staticmethod
    def _fingerprint(facts: dict) -> str:
        rounded = {
            k: (round(v, 2) if isinstance(v, float) else v)
            for k, v in facts.items()
            if k != "top_drivers"
        }
        rounded["_prompt_version"] = PROMPT_VERSION
        raw = json.dumps(rounded, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def has(self, key: str) -> bool:
        """Cheap existence check for UI hints -- doesn't verify the fingerprint,
        so use get() when the actual text needs to be trusted as current."""
        return key in self._data

    def get(self, key: str, facts: dict) -> str | None:
        entry = self._data.get(key)
        if entry and entry.get("fingerprint") == self._fingerprint(facts):
            return entry["text"]
        return None

    def set(self, key: str, facts: dict, text: str) -> None:
        self._data[key] = {
            "fingerprint": self._fingerprint(facts),
            "text": text,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }


def get_cached_insight(
    cache: InsightCache,
    daily_summary: pd.DataFrame,
    shap_df: pd.DataFrame | None,
    extreme_dates: set,
    full_year_baseline_mae: float,
    start_date: dt.date,
    end_date: dt.date,
) -> str | None:
    """Look up a period's insight without calling the API -- lets the caller
    decide whether to show existing text or offer a "generate" action."""
    facts = period_facts(
        daily_summary,
        shap_df,
        extreme_dates,
        full_year_baseline_mae,
        start_date,
        end_date,
    )
    return cache.get(period_key(start_date, end_date), facts)


def generate_and_cache_insight(
    cache: InsightCache,
    client: Groq,
    daily_summary: pd.DataFrame,
    shap_df: pd.DataFrame | None,
    extreme_dates: set,
    full_year_baseline_mae: float,
    start_date: dt.date,
    end_date: dt.date,
) -> str:
    """Actually calls the API for one period and saves the result. Only meant
    to run when a person is looking at this exact period -- see run_llm_insights
    for the small, deliberate set of periods primed ahead of time instead."""
    label = format_period_label(start_date, end_date)
    facts = period_facts(
        daily_summary,
        shap_df,
        extreme_dates,
        full_year_baseline_mae,
        start_date,
        end_date,
    )
    text = generate_insight(client, build_prompt(label, facts))
    cache.set(period_key(start_date, end_date), facts, text)
    cache.save()
    return text


def default_priming_periods(
    daily_summary: pd.DataFrame, top_n: int = TOP_N_DEFAULT
) -> list[tuple]:
    """The handful of views the app's own quick-select buttons point at, plus
    the top error days -- a small, bounded set worth generating ahead of time
    so the deployed app works without a live key for the common views. Every
    other date or range a person picks is generated live instead, on demand."""
    available_dates = sorted(daily_summary.index.date)
    most_recent, yesterday = available_dates[-1], available_dates[-2]
    last7, last30 = available_dates[-7:], available_dates[-30:]

    periods = [
        (most_recent, most_recent),
        (yesterday, yesterday),
        (last7[0], last7[-1]),
        (last30[0], last30[-1]),
        (available_dates[0], available_dates[-1]),
    ]
    for date in top_error_days(daily_summary, top_n).index:
        periods.append((date.date(), date.date()))
    return periods


def run_llm_insights(dry_run: bool = False, top_n: int = TOP_N_DEFAULT) -> InsightCache:
    """Primes the cache for the default views only -- not every day in the
    test year. Set dry_run=True to check the prompts without spending API calls."""
    predictions_df = pd.read_parquet(config.WEATHER_MODEL_PREDICTIONS_PATH)
    daily_summary = daily_error_summary(predictions_df)
    shap_df = load_shap_contributions()
    _, extreme_dates = compute_extreme_day_context()
    full_year_baseline_mae = float(daily_summary["mean_abs_error_mw"].mean())

    cache = InsightCache()
    client = None if dry_run else Groq(api_key=os.environ["GROQ_API_KEY"])

    for start_date, end_date in default_priming_periods(daily_summary, top_n):
        label = format_period_label(start_date, end_date)
        facts = period_facts(
            daily_summary,
            shap_df,
            extreme_dates,
            full_year_baseline_mae,
            start_date,
            end_date,
        )

        if cache.get(period_key(start_date, end_date), facts) is not None:
            print(f"already cached: {label}")
            continue

        if dry_run:
            print(f"[dry run] {label}\n{build_prompt(label, facts)}\n")
            continue

        text = generate_insight(client, build_prompt(label, facts))
        cache.set(period_key(start_date, end_date), facts, text)
        cache.save()
        print(f"generated: {label}")

    if not dry_run:
        print(f"cache saved to {config.LLM_INSIGHT_CACHE_PATH}")
    return cache


if __name__ == "__main__":
    run_llm_insights()
