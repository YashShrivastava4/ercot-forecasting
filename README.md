# ERCOT Coast Zone Weather-Aware Load Forecasting

Forecasts hourly electricity demand for ERCOT's Coast (Houston) weather zone,
and measures a specific, real failure mode: a calendar-only forecast model
gets worse on statistically extreme-temperature days, because it has no
concept of weather. A temperature-aware model is built to close that gap,
and SHAP is used to confirm the improvement is actually driven by
temperature, not coincidence.

## Result

*(fill in after running the full pipeline — this is the one line that matters)*

> Built a weather-aware electricity demand forecasting model on real ERCOT
> (Texas grid) data; identified that a calendar-only baseline underperformed
> by **[X]%** on statistically extreme temperature days, and a temperature-aware
> XGBoost model closed **[Y]%** of that gap specifically because of weather —
> isolated with a same-model no-weather ablation, not just SHAP ranking —
> with an LLM-generated plain-language explanation layer for the highest-error
> forecast days.

The comparison table backing this number (baseline, no-weather ablation, and
weather-aware, each split by normal vs. extreme days) is saved to
`outputs/comparison_table.csv` after running notebook 04.

## What you need to do

This section assumes no prior familiarity with any of the specific tools
below (`gridstatus`, `meteostat`, Groq). If you already know some of these,
skip ahead.

### 1. Set up the Python environment

```bash
cd project3-ercot-forecasting
python3 -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get a Groq API key (only needed for the LLM insight step)

1. Go to [console.groq.com](https://console.groq.com) and sign up (free tier
   is enough for this).
2. Create an API key from the dashboard.
3. Copy `.env.example` to a new file named `.env` in the project root.
4. Paste your key in: `GROQ_API_KEY=gsk_...`

You can skip this until step 5 below, everything before that doesn't need it.

### 3. Pull and join the data

```bash
python3 -m src.data_pipeline
```

This pulls ERCOT hourly load (via `gridstatus`, no account needed for the
first attempt) and Houston hourly weather (via `meteostat`, no account
needed at all), joins them, and saves
`data/processed/load_weather_joined.parquet`.

**This step needs real internet access to ERCOT's and meteostat's servers.**
It will not work in a network-sandboxed environment. Run it on a normal
machine or in Colab.

Read the printed output carefully the first time you run it — it tells you
which years actually came back, and flags any day that doesn't have exactly
24 hourly rows (expected only on the two DST changeover days each year). If
`gridstatus` fails to find data (ERCOT occasionally changes its archive page
layout), see the "If the data pull fails" section below.

### 4. Run the notebooks, in order

Open Jupyter (`jupyter notebook` or `jupyter lab` from the project root) and
run, top to bottom:

1. `notebooks/01_eda.ipynb` — look at the raw data, confirm the load vs.
   temperature relationship looks sensible before trusting any model built
   on top of it.
2. `notebooks/02_baseline_model.ipynb` — fits the calendar-only baseline.
   Takes a few minutes on multi-year hourly data, that's expected.
3. `notebooks/03_extreme_day_analysis.ipynb` — the core finding. Read the
   printed extreme-day count; if it's under roughly 15-20 days, widen
   `EXTREME_TEMP_PERCENTILE` in `src/config.py` from 5 to 10 and rerun.
4. `notebooks/04_weather_aware_model.ipynb` — fits the XGBoost model, runs
   SHAP, fits a second no-weather ablation model (same features minus
   temperature), and builds the final comparison table across all three
   models.

### 5. Generate the LLM explanations (needs the `.env` key from step 2)

```bash
python3 -m src.llm_insights
```

Saves `outputs/llm_explanations.csv`. To check the prompts look right before
spending real API calls, open a Python shell and run
`llm_insights.run_llm_insights(dry_run=True)` first.

### 6. (Optional, lowest priority) Run the Streamlit viewer

```bash
streamlit run app/streamlit_app.py
```

Only worth doing once steps 1-5 are done with time to spare.

### If the data pull fails

`gridstatus`'s Tier 1 method scrapes ERCOT's public load archive page. If
that page's layout has changed or a year isn't published yet, you'll see it
skipped in the printed output rather than a silent gap. Two fallbacks, in
order:

- **Tier 2:** `gridstatus.ercot_api`, ERCOT's official API. Needs a free
  ERCOT API account (username/password as env vars). Check `gridstatus`'s
  own README for the current method name, this shifts between library
  versions.
- **Tier 3:** manual download from ERCOT's Hourly Load Data Archives page,
  then load the file directly with pandas instead of `src/data_pipeline.py`.

## Project structure

```
project3-ercot-forecasting/
  data/
    raw/            pulled ERCOT + weather data, untouched
    processed/      joined, cleaned parquet + saved model predictions
  models/           saved baseline (pickle), weather-aware, and no-weather
                    ablation models (xgboost json)
  notebooks/        run in numeric order, see above
  src/
    config.py         paths and constants used everywhere else
    data_pipeline.py  pulls and joins ERCOT load + Houston weather, closes any
                      hourly gaps so lag features downstream stay accurate
    utils.py          shared metrics and leakage-safe extreme-day logic
    baseline_model.py calendar-only SARIMAX baseline
    weather_model.py  XGBoost weather-aware model + SHAP, plus a no-weather
                      ablation (same features, temperature removed) to isolate
                      what weather specifically contributes
    llm_insights.py   plain-English explanations for the worst forecast days
  app/
    streamlit_app.py  optional day-by-day viewer
  outputs/            comparison_table.csv, llm_explanations.csv
```

## Design decisions worth knowing before an interview

- **SARIMAX, not textbook seasonal SARIMA.** A seasonal ARIMA with the
  seasonal period set to 24 hours doesn't scale to several years of hourly
  data with statsmodels, the state-space fit is impractically slow. The
  baseline instead uses SARIMAX with a compact ARIMA(2,1,2) error structure
  plus Fourier terms for daily/annual cycles and day-of-week/holiday dummies
  as exogenous regressors — same calendar-only information, parameterized so
  it actually fits in a few minutes. This is a legitimate SARIMAX
  variant, not a shortcut that changes what's being tested.
- **No database.** Project 2 used Postgres because it needed to serve a
  chatbot. This is a dataframe-driven modeling workflow, a DB adds nothing
  here. Parquet files only.
- **A no-weather ablation model, not just SHAP, is what backs the "it's the
  weather" claim.** The baseline and the weather-aware model differ in more
  than one way — SARIMAX vs. XGBoost, and the weather-aware model also gets
  real recent-load lag features the baseline never sees. That's a legitimate
  design choice, but on its own it can't tell you whether an improvement over
  baseline came from switching model families or from temperature. The
  ablation (same XGBoost, same lag/rolling/calendar features, temperature
  removed) holds everything but weather constant, so the delta between it and
  the weather-aware model is the actual, isolated contribution of weather.
- **The joined data is reindexed onto a complete, gap-free hourly grid before
  any lag or rolling feature is built** (`src/data_pipeline.py`,
  `close_hourly_gaps`). Both ERCOT and meteostat can be missing individual
  hours; short gaps (up to 3 hours) are time-interpolated, longer ones are
  left as NaN and reported rather than guessed. This matters because
  `load_lag_1h` etc. are built with `.shift()`, which is positional — on an
  index with missing timestamps, "1 row back" silently stops meaning "1 hour
  back" near any gap. Making the grid complete first is what keeps that
  guarantee true everywhere.
- **Single Houston point as a proxy for the whole Coast zone**, and
  **industrial/petrochemical demand in that zone dilutes the temperature
  signal** relative to a purely residential zone. Both are real, known
  simplifications, not bugs — have them ready if asked.
- **All extreme-day and temperature-deviation thresholds are computed on
  training data only**, then applied as fixed numbers to the test set. This
  is checked in code (`src/utils.py`), not just asserted in this README.
- **Lag features go up to 168 hours (one week).** This is closer to a
  short-horizon / nowcasting setup than a true day-ahead forecast — worth
  being explicit about that distinction if it comes up, since it changes
  what the model is actually allowed to "see" before predicting.

## Known pitfalls this code explicitly handles

- DST transitions (23-hour and 25-hour days) — `gridstatus` handles this
  internally for the ERCOT side, `src/data_pipeline.py` still checks and
  prints every day that isn't exactly 24 hours, so this is visible rather
  than silently wrong.
- Missing individual hours (dropped NaN rows, a gap in the weather record) —
  the joined data is reindexed onto a complete hourly grid and short gaps are
  interpolated, so `.shift()`-based lag features never silently misalign
  around a missing hour. Longer gaps are left as NaN and reported, not guessed.
- Extreme-day and temperature-deviation leakage — thresholds and the
  day-of-year climatology are both computed from training data only.
- Rolling-average leakage — the 24h/168h rolling load features are shifted
  by one hour first, so the current hour's own value never leaks into its
  own feature.
