# Food CPI Agent Reference

This reference is for the code-execution sandbox. Assume the installed
`aieng.forecasting` package is available.

## Exact Imports

```python
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters.statcan import StatCanAdapter
from aieng.forecasting.evaluation import (
    BacktestSpec,
    ContinuousForecast,
    ForecastingTask,
    Prediction,
    STANDARD_QUANTILES,
    backtest,
)
from aieng.forecasting.methods import (
    DartsAutoARIMAPredictor,
    DartsLightGBMPredictor,
    DartsLinearRegressionPredictor,
    LastValuePredictor,
)
```

Use these imports first. Only call `help()` or `dir()` if one of these exact
imports fails in the installed package.

## Food CPI Series

```python
STATCAN_TABLE = "18-10-0004-11"

FOOD_CPI_SERIES = [
    (
        "cpi_food_canada",
        "Food",
        "CPI Food (overall), Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_bakery_cereal_canada",
        "Bakery and cereal products (excluding baby food)",
        "CPI Bakery and cereal products (excl. baby food), Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_dairy_eggs_canada",
        "Dairy products and eggs",
        "CPI Dairy products and eggs, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_fish_seafood_canada",
        "Fish, seafood and other marine products",
        "CPI Fish, seafood and other marine products, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_restaurants_canada",
        "Food purchased from restaurants",
        "CPI Food purchased from restaurants, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_fruit_preparations_nuts_canada",
        "Fruit, fruit preparations and nuts",
        "CPI Fruit, fruit preparations and nuts, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_meat_canada",
        "Meat",
        "CPI Meat, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_other_food_nonalcoholic_canada",
        "Other food products and non-alcoholic beverages",
        "CPI Other food and non-alcoholic beverages, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_vegetables_preparations_canada",
        "Vegetables and vegetable preparations",
        "CPI Vegetables and vegetable preparations, Canada (2002=100)",
        "Index 2002=100",
    ),
]
```

The `product_group` strings above must exactly match StatCan's
`Products and product groups` column.

## Build The Data Service

```python
def build_food_cpi_service(cache_dir="data/statcan"):
    service = DataService()
    for series_id, product_group, description, units in FOOD_CPI_SERIES:
        adapter = StatCanAdapter(
            table_id=STATCAN_TABLE,
            member_filter={
                "GEO": "Canada",
                "Products and product groups": product_group,
            },
            cache_dir=cache_dir,
        )
        service.register(
            series_id,
            adapter,
            SeriesMetadata(
                series_id=series_id,
                description=description,
                source="Statistics Canada",
                units=units,
                frequency="MS",
                table_id=STATCAN_TABLE,
            ),
        )
    return service
```

Do not use `stats_can.zip_table_to_dataframe` for the normal path.
`StatCanAdapter` downloads/reads the StatCan zip and returns canonical
`timestamp`, `value`, `released_at` columns.

## Build CFPR-Style Tasks

The canonical CFPR trajectory forecasts January through December of the
following calendar year from a July origin, so use horizons `6..17` with monthly
frequency.
When using the package evaluation harness, copy this helper shape exactly unless
the user asks for a different task definition.

```python
CFPR_HORIZONS = list(range(6, 18))

def build_food_cpi_tasks():
    return [
        ForecastingTask(
            task_id=f"{series_id.removeprefix('cpi_').removesuffix('_canada')}_cfpr",
            target_series_id=series_id,
            horizons=CFPR_HORIZONS,
            frequency="MS",
            description=f"{description}; Jan-Dec trajectory from a July origin.",
        )
        for series_id, _product_group, description, _units in FOOD_CPI_SERIES
    ]
```

Task IDs only need to be stable and readable. The `target_series_id` must match
the `DataService` registration key.

## Minimal Setup Script

Use this as the first code call when the user asks for an interactive CFPR-style
experiment. Replace `predictor = ...` with the predictor the user requested.
Keep setup and analysis together unless you have a specific reason to probe
separately.
Do not spend a separate code call just checking the latest available timestamp;
print any sanity checks from this same run.

```python
from datetime import datetime

from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters.statcan import StatCanAdapter
from aieng.forecasting.evaluation import BacktestSpec, ForecastingTask, backtest
from aieng.forecasting.methods import LastValuePredictor

# Paste STATCAN_TABLE, FOOD_CPI_SERIES, build_food_cpi_service, and
# build_food_cpi_tasks from this reference here.

service = build_food_cpi_service()
tasks = build_food_cpi_tasks()

predictor = LastValuePredictor()
spec = BacktestSpec(
    task=tasks[0],
    start=datetime(2021, 7, 1),
    end=datetime(2024, 7, 1),
    stride=12,
    warmup=24,
)
result = backtest(predictor=predictor, spec=spec, data_service=service)
print(result.mean_crps)
```

If the user asks to compare all nine categories, loop over `tasks` and run one
`BacktestSpec` per task, or use `MultiTargetBacktestSpec` / `multi_backtest` if
you have confirmed those imports are available.

## Evaluation Harness Contracts

`ForecastingTask` defines what to forecast:

```python
task = ForecastingTask(
    task_id="food_cpi_overall_cfpr",
    target_series_id="cpi_food_canada",
    horizons=list(range(6, 18)),
    frequency="MS",
    description="Canada CPI Food, Jan-Dec trajectory from a July origin.",
)
```

`BacktestSpec` defines which origins to evaluate. It requires `start < end`.
For a single-origin run, use a tiny interval that contains one origin:

```python
from datetime import timedelta

origin = datetime(2024, 7, 1)
single_origin_spec = BacktestSpec(
    task=task,
    start=origin,
    end=origin + timedelta(days=1),
    stride=1,
    warmup=24,
)
```

`backtest(...)` returns a `BacktestResult`, not a list. Forecasts are in
`result.predictions`; each prediction stores the target date in
`forecast_date` and the median/point forecast in `payload.point_forecast`.
`Prediction` does not store a `horizon` attribute. If you need a horizon label,
derive it from `task.horizons`, enumerate predictions in task order, or compute
it from `forecast_date - as_of` using the task frequency.

```python
result = backtest(predictor=predictor, spec=single_origin_spec, data_service=service)
forecast_df = pd.DataFrame(
    {
        "horizon": task.horizons[index],
        "timestamp": prediction.forecast_date,
        "value": prediction.payload.point_forecast,
    }
    for index, prediction in enumerate(result.predictions)
)
```

## Data Access Contracts

Use `DataService` for registered series and cutoff-safe history:

```python
history = service.get_series("cpi_food_canada", as_of=origin)
```

Within a `Predictor`, use the provided context:

```python
series = context.get_series(task.target_series_id)
```

If you need the full raw adapter output for analysis outside the harness, keep
the adapter object and call `adapter.fetch()`. `DataService` itself is the
registration/query layer, not a raw data-fetch API.

## Prediction Extraction

Use this shape whenever you need a tabular view of forecasts from package
predictors:

```python
def predictions_to_frame(result):
    return pd.DataFrame(
        {
            "as_of": prediction.as_of,
            "forecast_date": prediction.forecast_date,
            "point_forecast": prediction.payload.point_forecast,
            **{
                f"q{int(quantile * 100):02d}": value
                for quantile, value in prediction.payload.quantiles.items()
            },
        }
        for prediction in result.predictions
    )
```

If horizon labels are needed and the result comes from one task, add them from
the task definition:

```python
forecast_df = predictions_to_frame(result)
forecast_df["horizon"] = task.horizons[: len(forecast_df)]
```

## Reporting Tables

Use display formats that work with the standard installed stack:

```python
summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
# or
print(summary.to_csv(index=False))
```

## Plotting Forecasts

For plotting, build a DataFrame from `result.predictions` and plot
`forecast_date` on the x-axis. Include historical cutoff-safe observations when
useful.

```python
import matplotlib.pyplot as plt

history = service.get_series(task.target_series_id, as_of=origin)
forecast_df = predictions_to_frame(result)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(history["timestamp"], history["value"], label="History", color="black")
ax.plot(
    forecast_df["forecast_date"],
    forecast_df["point_forecast"],
    label="Point forecast",
    color="tab:blue",
)
ax.fill_between(
    forecast_df["forecast_date"],
    forecast_df["q05"],
    forecast_df["q95"],
    color="tab:blue",
    alpha=0.15,
    label="90% interval",
)
ax.set_title(task.description)
ax.set_xlabel("Date")
ax.set_ylabel("CPI index")
ax.legend()
fig.tight_layout()
fig.savefig("forecast.png", dpi=150)
print("Saved forecast.png")
```

## Reporting Across Multiple Categories

If the user asks about more than one food category (for example the full CFPR
report across all nine series), present the result as a clear per-category
view: one row per category in a summary table, or per-category forecast tables
when monthly paths are requested. Do not silently collapse multiple categories
into a single unlabeled forecast list; group the output by `series_id` or
category name so each forecast is attributable to its target series.

## CFPR Framing

The Canada's Food Price Report headline is often stated as an average-over-average
YoY percentage change:

```text
mean(CPI in Jan-Dec of Y+1) / mean(CPI in Jan-Dec of Y) - 1
```

The evaluation harness scores the monthly CPI level forecasts directly with CRPS.
The avg/avg YoY transformation is downstream analysis, not the primary forecast
payload.

## API Contracts And Guardrails

- Import only from `food_price_forecasting.*` in the sandbox if the user has
  confirmed that implementation package is installed.
- The baseline export is `LastValuePredictor`, not `NaivePredictor` or
  `SeasonalNaivePredictor`.
- `StatCanAdapter` requires `table_id` and `member_filter`.
- `member_filter` maps table column names to values. For food CPI, use
  `{"GEO": "Canada", "Products and product groups": product_group}`.
- `ForecastingTask` requires `task_id`, `target_series_id`, `horizons`,
  `frequency`, and `description`.
- `Prediction` stores forecast values under `prediction.payload`; do not assume
  payload fields are copied onto the wrapper object.
- `Prediction` has no `horizon` attribute; derive horizon labels from the task or
  dates when needed.
