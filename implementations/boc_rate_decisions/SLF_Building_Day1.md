# SLF Building Day 1

## Purpose

This note summarizes the work completed in the `implementations/boc_rate_decisions` area to add **StatCan GDP** as an additional macro covariate, update the exploratory notebook, and validate that the new data path works end to end.

## Requested change

The task was to extend the Bank of Canada rate-decision use case so that:

1. GDP from StatCan is available through the existing BoC data-service path.
2. Notebook `01_boc_data_exploration.ipynb` reflects GDP as an additional macro covariate.
3. Section `6. Macro covariates` includes GDP analysis and a GDP visualization.
4. Changes remain minimal and reuse the repo's existing libraries and patterns.

## What was changed

### 1. Added a GDP series to the BoC data service

File: `implementations/boc_rate_decisions/data.py`

Changes made:

- Added a new canonical series id:
  - `GDP_SERIES_ID = "canada_real_gdp_all_industries"`
- Added a new StatCan table constant:
  - `GDP_TABLE_ID = "36-10-0434-01"`
- Registered the GDP series inside `build_boc_service(...)` using the existing `StatCanAdapter`.
- Exposed both new constants through `__all__`.
- Updated the module-level description so it now lists GDP alongside CPI, unemployment, and bond yields.
- Adjusted the wording to avoid overstating current model usage: the service now describes these series as available for **shared notebook exploration and predictor context**, rather than implying GDP is already part of the existing baseline feature set.

Series registration details:

- Table: `36-10-0434-01`
- Geography filter: `Canada`
- Seasonal adjustment: `Seasonally adjusted at annual rates`
- Price basis: `Chained (2017) dollars`
- Industry slice: `All industries [T001]`
- Frequency recorded in metadata: monthly (`MS`)

This keeps GDP wired through exactly the same `DataService` abstraction that the notebook already uses for the other macro series.

### 2. Updated the BoC cache-population script

File: `scripts/fetch_boc.py`

Changes made:

- Added the GDP table to the script documentation.
- Added the normalized cached zip name `36100434-eng.zip` to `_TABLE_ZIPS`.

Effect:

- Running `uv run python scripts/fetch_boc.py` now populates the StatCan GDP cache along with the rate, yield, and CPI tables.

### 3. Updated BoC use-case documentation

File: `implementations/boc_rate_decisions/README.md`

Changes made:

- Added GDP to the `Data` table.
- Described it as:
  - `Real GDP, all industries`
  - `StatCan 36-10-0434-01`
  - monthly, chained 2017 dollars, seasonally adjusted annual rates
- Updated the `fetch_boc.py` cache command comment so it now says the script fetches `rate, 2yr yield, CPI, GDP, unemployment`.

### 4. Updated the BoC exploration notebook

File: `implementations/boc_rate_decisions/01_boc_data_exploration.ipynb`

Changes were intentionally limited to the cells directly affected by the GDP addition.

#### Section 1 setup text

Updated the notebook setup text so it now says the use case has **four** macro covariates instead of three:

- 2-year GoC yield
- CPI
- GDP
- unemployment

#### Setup/import code cell

Updated the import list so the notebook pulls in `GDP_SERIES_ID` from `boc_rate_decisions.data`.

#### Section 5 cutoff-discipline text

Updated the monthly-release discussion so that the approximate `released_at` handling now explicitly includes **GDP** as well as CPI.

#### Section 6 macro-covariates text

Expanded the macro-covariates explanation from three series to four.

The notebook now describes GDP as:

- a broad activity signal
- useful for distinguishing overheating from soft-landing or recessionary regimes
- informative about whether restrictive policy is biting or whether activity is re-accelerating

#### Section 6 chart cell

Updated the plotting cell so it now:

- fetches `gdp_df = svc.get_series(GDP_SERIES_ID, as_of=_as_of)`
- computes `gdp_yoy`
- expands the figure from **3 rows** to **4 rows**
- adds a fourth subplot for real GDP YoY growth

The four chart panels are now:

1. Policy rate vs 2-year GoC yield
2. CPI inflation vs 2% target
3. Unemployment rate
4. Real GDP growth (all industries, chained 2017 dollars)

## Validation and checks performed

Validation was done immediately after the edits, using the narrowest practical checks.

### 1. Verified the GDP table and filters before editing

Actions performed:

- Inspected the installed `stats_can` package to confirm the available helper functions.
- Downloaded and inspected candidate StatCan GDP table `36-10-0434-01`.
- Examined its columns and categorical values to identify the exact filter combination needed for a single Canada headline GDP series.

This confirmed that the table and member filters were compatible with the repo's existing `StatCanAdapter` design.

### 2. Notebook execution checks

Notebook: `implementations/boc_rate_decisions/01_boc_data_exploration.ipynb`

Executed cells after the edit:

- Code cell 3: setup / `build_boc_service(...)`
- Code cell 13: macro covariate chart

Observed results:

- The setup cell executed successfully.
- During the run, the GDP cache file `36100434-eng.zip` was downloaded successfully.
- The notebook kernel now contains `GDP_SERIES_ID`, `gdp_df`, and `gdp_yoy`.
- The macro-covariate figure rendered successfully with the new fourth GDP panel.

### 3. Plain Python integration check outside the notebook

Ran a direct Python check from the workspace root with the local repo paths inserted into `sys.path`, then:

- imported `GDP_SERIES_ID` and `build_boc_service`
- built the BoC service against `data/statcan` and `data/fred`
- requested the GDP series from the service

Observed result:

- the GDP series loaded successfully
- the returned data spanned from `1997-01-01` through `2026-05-01`

This confirmed that the change was not only a notebook edit; the BoC data-service API itself now serves GDP correctly.

## Current scope boundaries

What this work did **not** change:

- No predictor logic was updated.
- The logistic baseline still uses its existing feature set.
- The agent prompt/context builders were not modified.
- No evaluation specs were changed.

In other words, GDP is now available in the BoC service and visible in the exploration notebook, but it has not yet been added as an active model feature in the BoC predictors.


## End state

At the end of this work:

- the BoC data service exposes a reusable StatCan GDP series
- the BoC cache script fetches the GDP table
- the BoC README documents GDP
- the BoC data exploration notebook explains GDP and plots it
- notebook execution and direct service access both succeeded
