"""Streamlit viewer: pick a date and see actual vs predicted load for that
day, plus an LLM insight -- already generated if this view was primed ahead
of time, or generated live on request otherwise. Also supports a multi-day
view (7/30 days, full year) and a top-error-days view, each with their own
insight for that exact period.

Run with: streamlit run app/streamlit_app.py
"""

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from src import config
from src.llm_insights import (
    InsightCache,
    compute_extreme_day_context,
    daily_error_summary,
    generate_and_cache_insight,
    get_cached_insight,
    load_shap_contributions,
    period_key,
    section_to_html,
    split_insight_sections,
    top_error_days,
)

load_dotenv()

st.set_page_config(page_title="ERCOT Coast Zone Forecast Viewer", layout="wide")

ACTUAL_COLOR = "#7FB9EE"
PREDICTED_COLOR = "#1B4FA0"

# card colors use Streamlit's own theme variables (var(--...)) plus low-opacity
# rgba tints, so the whole page follows the user's light/dark theme automatically
CARD_STYLE = """
<style>
.metric-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 12px;
    padding: 18px 20px;
    height: 100%;
}
.metric-icon {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    margin-bottom: 10px;
}
.metric-label {
    color: var(--text-color);
    opacity: 0.65;
    font-size: 13px;
    margin-bottom: 2px;
}
.metric-value {
    color: var(--text-color);
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 2px;
}
.metric-caption {
    color: var(--text-color);
    opacity: 0.55;
    font-size: 12px;
}
.explanation-card {
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.30);
    border-radius: 12px;
    padding: 18px 20px;
}
.insight-section {
    margin-bottom: 14px;
}
.insight-section:last-child {
    margin-bottom: 0;
}
.insight-label {
    font-weight: 700;
    font-size: 13px;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.insight-model-label { color: #10B981; }
.insight-business-label { color: #3B82F6; }
.insight-body {
    color: var(--text-color);
    font-size: 14px;
    line-height: 1.6;
}
.insight-body p {
    margin: 0 0 8px 0;
}
.insight-body p:last-child {
    margin-bottom: 0;
}
.insight-list {
    margin: 0;
    padding-left: 20px;
}
.insight-list li {
    margin-bottom: 6px;
}
.insight-list li:last-child {
    margin-bottom: 0;
}
.insight-placeholder {
    color: var(--text-color);
    opacity: 0.7;
    font-size: 14px;
    margin-bottom: 10px;
}
.error-day-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.error-day-meta {
    color: var(--text-color);
    opacity: 0.65;
    font-size: 13px;
}
</style>
"""
st.markdown(CARD_STYLE, unsafe_allow_html=True)

# rgba tints for the metric icon circles, chosen to read well on both themes
ICON_BLUE = "rgba(59, 130, 246, 0.16)"
ICON_GREEN = "rgba(16, 185, 129, 0.16)"
ICON_PURPLE = "rgba(168, 85, 247, 0.16)"


def metric_card(icon: str, icon_bg: str, label: str, value: str, caption: str) -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-icon" style="background:{icon_bg}">{icon}</div>'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-caption">{caption}</div>'
        "</div>"
    )


def themed_line_chart(x_actual, y_actual, x_pred, y_pred, x_title: str, y_title: str) -> go.Figure:
    """A transparent-background chart so it blends into whichever Streamlit
    theme is active -- the "streamlit" chart theme (the st.plotly_chart default)
    fills in the correct text/gridline colors for light or dark mode."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_actual, y=y_actual, name="Actual Load (MW)",
                              line=dict(color=ACTUAL_COLOR, width=2.5)))
    fig.add_trace(go.Scatter(x=x_pred, y=y_pred, name="Predicted Load (MW)",
                              line=dict(color=PREDICTED_COLOR, width=2.5)))
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=x_title,
        yaxis_title=y_title,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_parquet(config.WEATHER_MODEL_PREDICTIONS_PATH)


@st.cache_data
def load_daily_summary(_predictions: pd.DataFrame) -> pd.DataFrame:
    return daily_error_summary(_predictions)


@st.cache_data
def load_shap() -> pd.DataFrame | None:
    return load_shap_contributions()


@st.cache_data
def load_extreme_dates() -> set:
    _, extreme_dates = compute_extreme_day_context()
    return extreme_dates


@st.cache_resource
def get_groq_client() -> Groq | None:
    import os
    api_key = os.environ.get("GROQ_API_KEY")
    return Groq(api_key=api_key) if api_key else None


def render_insight_card(model_text: str, business_text: str) -> None:
    body = (
        '<div class="insight-section">'
        '<div class="insight-label insight-model-label">\U0001f9e0 Model insight</div>'
        f'<div class="insight-body">{section_to_html(model_text)}</div></div>'
    )
    if business_text:
        body += (
            '<div class="insight-section">'
            '<div class="insight-label insight-business-label">\U0001f3e2 Business insight</div>'
            f'<div class="insight-body">{section_to_html(business_text)}</div></div>'
        )
    st.markdown(f'<div class="explanation-card">{body}</div>', unsafe_allow_html=True)


def show_insight(start_date: dt.date, end_date: dt.date, key_suffix: str) -> None:
    """Shows a cached insight if one exists for this exact period, otherwise
    offers to generate one live -- so every day or range can get a real
    explanation, not just the handful primed ahead of time."""
    cached = get_cached_insight(
        st.session_state.insight_cache, daily_summary, shap_df, extreme_dates,
        full_year_baseline_mae, start_date, end_date,
    )

    st.write("")
    if cached is not None:
        model_text, business_text = split_insight_sections(cached)
        render_insight_card(model_text, business_text)
        return

    client = get_groq_client()

    def _generate():
        with st.spinner("Generating insight..."):
            generate_and_cache_insight(
                st.session_state.insight_cache, client, daily_summary, shap_df,
                extreme_dates, full_year_baseline_mae, start_date, end_date,
            )

    st.markdown(
        '<div class="explanation-card">'
        '<div class="insight-label insight-model-label">\u2728 AI Insight</div>'
        '<div class="insight-placeholder">No insight has been generated for this '
        "view yet.</div></div>",
        unsafe_allow_html=True,
    )
    if client is not None:
        st.button("Generate insight for this view", key=f"gen_{key_suffix}", on_click=_generate)
    else:
        st.caption("Set GROQ_API_KEY to generate a new insight live for views that aren't already cached.")


if not config.WEATHER_MODEL_PREDICTIONS_PATH.exists():
    st.warning("No predictions found yet, run `src/weather_model.py` first.")
    st.stop()

predictions = load_predictions()
daily_summary = load_daily_summary(predictions)
shap_df = load_shap()
extreme_dates = load_extreme_dates()
full_year_baseline_mae = float(daily_summary["mean_abs_error_mw"].mean())

if "insight_cache" not in st.session_state:
    st.session_state.insight_cache = InsightCache()

available_dates = sorted(set(predictions.index.date))
most_recent = available_dates[-1]
full_year_days = len(available_dates)
top10 = top_error_days(daily_summary, top_n=10)

if "mode" not in st.session_state:
    st.session_state.mode = "single"
if "single_date" not in st.session_state:
    st.session_state.single_date = most_recent
if "range_days" not in st.session_state:
    st.session_state.range_days = 7


def go_single(date: dt.date) -> None:
    st.session_state.mode = "single"
    st.session_state.single_date = date


def go_range(days: int) -> None:
    st.session_state.mode = "range"
    st.session_state.range_days = days


def go_top_errors() -> None:
    st.session_state.mode = "top_errors"


with st.sidebar:
    st.markdown("### \U0001f4c8 ERCOT Coast Zone\nForecast Viewer")

    st.markdown("**Pick a date**")
    picked = st.date_input(
        "Pick a date",
        value=st.session_state.single_date,
        min_value=available_dates[0],
        max_value=available_dates[-1],
        label_visibility="collapsed",
    )
    if picked != st.session_state.single_date:
        go_single(picked)

    st.markdown("**Quick select**")

    def qbtn(label: str, active: bool, on_click) -> None:
        st.button(
            label,
            key=f"qsel_{label}",
            use_container_width=True,
            type="primary" if active else "secondary",
            on_click=on_click,
        )

    qbtn("Today", st.session_state.mode == "single" and st.session_state.single_date == most_recent,
         lambda: go_single(most_recent))
    qbtn("Yesterday",
         st.session_state.mode == "single" and st.session_state.single_date == available_dates[-2],
         lambda: go_single(available_dates[-2]))
    qbtn("Last 7 days", st.session_state.mode == "range" and st.session_state.range_days == 7,
         lambda: go_range(7))
    qbtn("Last 30 days", st.session_state.mode == "range" and st.session_state.range_days == 30,
         lambda: go_range(30))
    qbtn("Full year", st.session_state.mode == "range" and st.session_state.range_days == full_year_days,
         lambda: go_range(full_year_days))
    qbtn("Top error days (10)", st.session_state.mode == "top_errors", go_top_errors)

    st.caption(
        "\"Today\" and \"Yesterday\" point at the most recent dates in the "
        "2025 held-out test year, since this viewer only reads saved model "
        "output, not live data."
    )

    st.info(
        "Every view here -- a single day, a date range, or a top-error day -- "
        "gets its own model insight and business insight. Common views are "
        "pre-generated; anything else generates live the first time you look at it."
    )


# ---- single day view ----
if st.session_state.mode == "single":
    picked_date = st.session_state.single_date
    day_data = predictions[predictions.index.date == picked_date]

    st.title("Actual vs Predicted Load")
    st.caption("Explore the weather-aware forecast performance for any day in the 2025 test year.")

    if day_data.empty:
        st.info("No data for this date, pick another.")
        st.stop()

    mean_abs_error = (day_data["actual_mw"] - day_data["predicted_mw"]).abs().mean()
    avg_temp_deviation = day_data["temp_deviation"].mean()
    actual_avg = day_data["actual_mw"].mean()
    predicted_avg = day_data["predicted_mw"].mean()
    pct_error = (mean_abs_error / actual_avg) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("\u3030\ufe0f", ICON_BLUE, "Mean Absolute Error", f"{mean_abs_error:.0f} MW", f"{pct_error:.1f}% of actual demand"), unsafe_allow_html=True)
    c2.markdown(metric_card("\U0001f321\ufe0f", ICON_BLUE, "Avg Temp Deviation", f"{avg_temp_deviation:+.1f} \u00b0C", "Compared to historical norm"), unsafe_allow_html=True)
    c3.markdown(metric_card("\U0001f4c8", ICON_GREEN, "Actual Avg Load", f"{actual_avg/1000:.2f}k MW", "Average hourly actual load"), unsafe_allow_html=True)
    c4.markdown(metric_card("\u22ef", ICON_PURPLE, "Predicted Avg Load", f"{predicted_avg/1000:.2f}k MW", "Average hourly predicted load"), unsafe_allow_html=True)

    st.write("")
    st.markdown(f"**Actual vs predicted load \u2014 {picked_date}**")

    fig = themed_line_chart(day_data.index, day_data["actual_mw"], day_data.index, day_data["predicted_mw"], "Time", "MW")
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    show_insight(picked_date, picked_date, key_suffix=f"single_{picked_date}")

# ---- last 7 / 30 days / full year view ----
elif st.session_state.mode == "range":
    n = st.session_state.range_days
    is_full_year = n >= full_year_days
    range_dates = available_dates[-n:]
    range_daily = daily_summary.loc[
        (daily_summary.index.date >= range_dates[0]) & (daily_summary.index.date <= range_dates[-1])
    ]
    period_phrase = "the full 2025 test year" if is_full_year else f"the last {n} days of the 2025 test year"
    chart_phrase = "full test year" if is_full_year else f"last {n} days"

    st.title("Actual vs Predicted Load")
    st.caption(f"Daily averages over {period_phrase}.")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("\u3030\ufe0f", ICON_BLUE, "Avg Daily MAE", f"{range_daily['mean_abs_error_mw'].mean():.0f} MW", f"{range_daily['error_pct_avg'].mean():.1f}% of actual demand"), unsafe_allow_html=True)
    c2.markdown(metric_card("\U0001f321\ufe0f", ICON_BLUE, "Avg Temp Deviation", f"{range_daily['temp_deviation_avg'].mean():+.1f} \u00b0C", "Compared to historical norm"), unsafe_allow_html=True)
    c3.markdown(metric_card("\U0001f4c8", ICON_GREEN, "Actual Avg Load", f"{range_daily['actual_mw_avg'].mean()/1000:.2f}k MW", "Average hourly actual load"), unsafe_allow_html=True)
    c4.markdown(metric_card("\u22ef", ICON_PURPLE, "Predicted Avg Load", f"{range_daily['predicted_mw_avg'].mean()/1000:.2f}k MW", "Average hourly predicted load"), unsafe_allow_html=True)

    st.write("")
    st.markdown(f"**Daily actual vs predicted load \u2014 {chart_phrase}**")

    fig = themed_line_chart(range_daily.index, range_daily["actual_mw_avg"], range_daily.index, range_daily["predicted_mw_avg"], "Date", "MW (daily avg)")
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    show_insight(range_dates[0], range_dates[-1], key_suffix=f"range_{range_dates[0]}_{range_dates[-1]}")

    flagged = [d for d in range_dates if st.session_state.insight_cache.has(period_key(d, d))]
    if flagged:
        st.write("")
        st.markdown(f"**{len(flagged)} day(s) in this range already have a generated insight:**")
        for d in flagged:
            cols = st.columns([4, 1])
            cols[0].markdown(f"- **{d}**")
            cols[1].button("View", key=f"range_view_{d}", use_container_width=True, on_click=go_single, args=(d,))

# ---- top error days view ----
else:
    st.title("Top 10 Highest-Error Days")
    st.caption(
        "The days the weather-aware model's forecast missed by the widest "
        "margin in the 2025 test year. Open a day to see its full model and "
        "business insight."
    )

    for date, row in top10.iterrows():
        d = date.date()
        has_insight = st.session_state.insight_cache.has(period_key(d, d))
        cols = st.columns([3, 1])
        with cols[0]:
            badge = '<div class="error-day-meta" style="margin-top:6px">\u2728 Insight ready</div>' if has_insight else ""
            st.markdown(
                '<div class="error-day-card">'
                f"<b>{date.strftime('%B %d, %Y')}</b>"
                '<div class="error-day-meta">'
                f"MAE {row['mean_abs_error_mw']:.0f} MW ({row['error_pct_avg']:.1f}% of actual), "
                f"temp deviation {row['temp_deviation_avg']:+.1f} \u00b0C</div>"
                f"{badge}</div>",
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.button("View day", key=f"view_{d}", use_container_width=True, on_click=go_single, args=(d,))
