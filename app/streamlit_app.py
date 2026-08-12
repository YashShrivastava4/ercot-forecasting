"""Optional minimal viewer: pick a date, see actual vs predicted load for that
day, and the LLM explanation if that day was one of the highest-error days.

Run with: streamlit run app/streamlit_app.py
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config

st.set_page_config(page_title="ERCOT Coast Zone Forecast Viewer", layout="centered")
st.title("ERCOT Coast Zone Forecast Viewer")


@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_parquet(config.WEATHER_MODEL_PREDICTIONS_PATH)


@st.cache_data
def load_explanations() -> pd.DataFrame:
    if not config.LLM_EXPLANATIONS_PATH.exists():
        return pd.DataFrame()
    explanations = pd.read_csv(config.LLM_EXPLANATIONS_PATH, index_col="date", parse_dates=True)
    return explanations


if not config.WEATHER_MODEL_PREDICTIONS_PATH.exists():
    st.warning("No predictions found yet, run `src/weather_model.py` first.")
    st.stop()

predictions = load_predictions()
explanations = load_explanations()

available_dates = sorted(set(predictions.index.date))
picked_date = st.date_input(
    "Pick a date",
    value=available_dates[0],
    min_value=available_dates[0],
    max_value=available_dates[-1],
)

day_data = predictions[predictions.index.date == picked_date]

if day_data.empty:
    st.info("No data for this date, pick another.")
    st.stop()

fig = px.line(
    day_data,
    x=day_data.index,
    y=["actual_mw", "predicted_mw"],
    labels={"value": "MW", "index": "Hour", "variable": "Series"},
    title=f"Actual vs predicted load — {picked_date}",
)
st.plotly_chart(fig, use_container_width=True)

mean_abs_error = (day_data["actual_mw"] - day_data["predicted_mw"]).abs().mean()
avg_temp_deviation = day_data["temp_deviation"].mean()

col1, col2 = st.columns(2)
col1.metric("Mean absolute error", f"{mean_abs_error:.0f} MW")
col2.metric("Avg temp deviation", f"{avg_temp_deviation:+.1f} °C")

st.subheader("LLM explanation")
if explanations.empty:
    st.info("No explanations generated yet, run `src/llm_insights.py` first.")
elif pd.Timestamp(picked_date) in explanations.index:
    st.write(explanations.loc[pd.Timestamp(picked_date), "explanation"])
else:
    st.info("This day wasn't one of the highest-error days, so no explanation was generated for it.")
