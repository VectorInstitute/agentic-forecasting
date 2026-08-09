# Credit Driver Analysis — Code Patterns

These are working, copy-pasteable patterns for interpreting recent BAA10Ycovariates. 
Paste the relevant block into your code execution cell and adaptas needed.

---

## Pattern 1: Summarize recent driver movements
Classify the actual payload series by transformation. 
Do not place exportedPython objects such as DEFAULT_COVARIATE_SERIES_IDS, service builders, targetdefinitions, or registry names in these sets—they are not observations.



```python
import io
import numpy as np
import pandas as pd

# Assume `payload` and `analysis_window` were defined by the earlier skill.
market_level_series = {
    "vix_level_l1b",
    "ust10y_level_l1b",
    "ust2y10y_spread_l1b",
}

# These payload fields contain one-business-day log returns. Their cumulative
# window movement is the sum of log returns.
log_return_series = {
    "vix_log_ret_1b_l1b",
    "dollar_index_log_ret_1b_l1b",
    "gold_log_ret_1b_l1b",
    "nasdaq_log_ret_1b_l1b",
    "oil_log_ret_1b_l1b",
}

# These fields already contain one-business-day changes. Sum them across the
# window; do not difference them again.
change_series = {
    "hyoas_observed_change_1b_bps_l1b",
    "hyoas_hyg_dgs3_proxy_change_1b_bps_l1b",
}

# These slow or release-based series are context only. Repeated daily values
# from business-day expansion do not represent new releases.
slow_macro_series = {
    "fed_funds_level_l1b",
    "unemployment_rate_l1b",
    "cpi_mom_logdiff_l1b",
}

covariate_history = dict(payload.get("covariate_history", {}))

# Observed and proxy HYOAS describe the same credit factor. Prefer observed
# HYOAS whenever it is present and has data.
observed_hyoas = "hyoas_observed_change_1b_bps_l1b"
proxy_hyoas = "hyoas_hyg_dgs3_proxy_change_1b_bps_l1b"
if covariate_history.get(observed_hyoas):
    covariate_history.pop(proxy_hyoas, None)

# VIX return and VIX level describe the same volatility factor. Prefer the
# return as the scored signal and retain the level as qualitative context.
has_vix_return = bool(covariate_history.get("vix_log_ret_1b_l1b"))

rows = []

for series_id, records in covariate_history.items():
    series = pd.DataFrame(records)
    if not {"date", "value"}.issubset(series.columns):
        continue

    series["date"] = pd.to_datetime(series["date"], errors="coerce")
    series["value"] = pd.to_numeric(series["value"], errors="coerce")
    series = (
        series.dropna(subset=["date", "value"])
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .tail(analysis_window)
    )
    if len(series) < 2:
        continue

    values = series["value"].astype(float)
    latest_value = float(values.iloc[-1])
    previous_distinct_value = np.nan
    scorable = True

    if series_id in market_level_series:
        component_moves = values.diff().dropna()
        net_move = float(values.iloc[-1] - values.iloc[0])
        treatment = "level_change"

        if series_id == "vix_level_l1b" and has_vix_return:
            scorable = False
            treatment = "level_context_vix_return_preferred"

    elif series_id in log_return_series:
        component_moves = values
        net_move = float(values.sum())
        treatment = "cumulative_log_return"

    elif series_id in change_series:
        component_moves = values
        net_move = float(values.sum())
        treatment = "cumulative_change"

    elif series_id in slow_macro_series:
        # Collapse consecutive repeated values before comparing releases.
        distinct_values = values[values.ne(values.shift())]
        previous_distinct_value = (
            float(distinct_values.iloc[-2])
            if len(distinct_values) >= 2
            else np.nan
        )
        net_move = (
            latest_value - previous_distinct_value
            if np.isfinite(previous_distinct_value)
            else 0.0
        )
        component_moves = pd.Series(dtype=float)
        treatment = "macro_context_only"
        scorable = False

    else:
        # Unknown transformations must not be assigned a mechanical signal.
        component_moves = pd.Series(dtype=float)
        net_move = float(values.iloc[-1] - values.iloc[0])
        treatment = "unclassified_context_only"
        scorable = False

    move_z = np.nan
    if scorable and len(component_moves) >= 2:
        component_std = float(component_moves.std(ddof=1))
        scale = component_std * np.sqrt(len(component_moves))
        if np.isfinite(scale) and scale > 1e-12:
            move_z = float(net_move / scale)

    rows.append(
        {
            "series_id": series_id,
            "observations": int(len(series)),
            "treatment": treatment,
            "latest_value": latest_value,
            "previous_distinct_value": previous_distinct_value,
            "net_move": net_move,
            "move_z": move_z,
            "scorable": scorable,
        }
    )

driver_summary = pd.DataFrame(rows)
if not driver_summary.empty:
    driver_summary = driver_summary.sort_values(
        "move_z",
        key=lambda values: values.abs().fillna(-1.0),
        ascending=False,
    ).reset_index(drop=True)

print(driver_summary.to_string(index=False))
```
move_z is a scale-adjusted diagnostic for comparing active market signalswith different units. 
It is not a formal independent-sample test. 
The actualunits and transformations in payload["covariate_data_dictionary"] remain thesource of truth.



---

## Pattern 2: Translate movements into credit-spread evidence

Positive credit_score means widening evidence. 
Negative credit_score meanstightening evidence. 
Conditional market series and slow macro series remain indriver_context; 
they do not receive a fixed mechanical direction.

```python
# direction = +1: a positive move supports widening
# direction = -1: a positive move supports tightening

driver_rules = {
    "hyoas_observed_change_1b_bps_l1b": (+1, 3.0),
    "hyoas_hyg_dgs3_proxy_change_1b_bps_l1b": (+1, 2.0),
    "vix_log_ret_1b_l1b": (+1, 2.0),
    "vix_level_l1b": (+1, 2.0),  # fallback if VIX return is unavailable
    "nasdaq_log_ret_1b_l1b": (-1, 2.0),
    "dollar_index_log_ret_1b_l1b": (+1, 1.0),
    "oil_log_ret_1b_l1b": (-1, 1.0),
}

evidence_rows = []
context_rows = []

for row in driver_summary.to_dict("records"):
    rule = driver_rules.get(row["series_id"])
    move_z = row["move_z"]

    if (
        not row["scorable"]
        or rule is None
        or not np.isfinite(move_z)
    ):
        context_rows.append(row)
        continue

    direction, weight = rule
    credit_score = float(direction * move_z)

    if credit_score > 0.75:
        signal = "widening"
    elif credit_score < -0.75:
        signal = "tightening"
    else:
        signal = "neutral"

    evidence_rows.append(
        {
            **row,
            "credit_score": credit_score,
            "weight": weight,
            "signal": signal,
        }
    )

driver_evidence = pd.DataFrame(evidence_rows)
driver_context = pd.DataFrame(context_rows)

print("ACTIVE DRIVER EVIDENCE")
print(
    driver_evidence.to_string(index=False)
    if not driver_evidence.empty
    else "No scorable driver evidence"
)
print("\nCONTEXT-ONLY SERIES")
print(
    driver_context.to_string(index=False)
    if not driver_context.empty
    else "No context-only series"
)
```
Observed HYOAS is the strongest and most direct credit signal. 
Its proxy is afallback, not independent evidence. 
VIX and NASDAQ are primary risk-sentimentsignals. Dollar and oil are secondary signals. 
Treasury yields, the 2s10scurve, gold, Fed funds, unemployment, and CPI require regime or releasecontext, so do not force them into widening or tightening scores.
---

## Pattern 3: Combine widening and tightening evidence

Combine the usable signals into a conservative driver conclusion. 
This resultmay support or challenge the model forecast, but it must not mechanicallyreplace the point forecast or quantiles.



```python
if driver_evidence.empty:
    overall_signal = "neutral"
    confidence = "low"
    combined_score = 0.0
    widening_evidence = []
    tightening_evidence = []
else:
    bounded_scores = driver_evidence["credit_score"].clip(-3.0, 3.0)
    weights = driver_evidence["weight"].astype(float)
    combined_score = float(np.average(bounded_scores, weights=weights))

    if combined_score > 0.50:
        overall_signal = "widening"
    elif combined_score < -0.50:
        overall_signal = "tightening"
    else:
        overall_signal = "mixed"

    directional = driver_evidence[
        driver_evidence["signal"].isin(["widening", "tightening"])
    ].copy()

    if directional.empty:
        confidence = "low"
    else:
        widening_weight = float(
            directional.loc[
                directional["signal"] == "widening", "weight"
            ].sum()
        )
        tightening_weight = float(
            directional.loc[
                directional["signal"] == "tightening", "weight"
            ].sum()
        )
        total_directional_weight = widening_weight + tightening_weight
        agreement = (
            max(widening_weight, tightening_weight) / total_directional_weight
            if total_directional_weight > 0
            else 0.0
        )

        if len(directional) >= 2 and agreement >= 0.75:
            confidence = "high"
        elif len(directional) >= 2:
            confidence = "medium"
        else:
            confidence = "low"

    ranked = driver_evidence.assign(
        strength=(
            driver_evidence["credit_score"].abs()
            * driver_evidence["weight"]
        )
    ).sort_values("strength", ascending=False)

    widening_evidence = ranked.loc[
        ranked["signal"] == "widening", "series_id"
    ].head(3).tolist()
    tightening_evidence = ranked.loc[
        ranked["signal"] == "tightening", "series_id"
    ].head(3).tolist()

print(f"DRIVER_SIGNAL: {overall_signal}")
print(f"COMBINED_SCORE: {combined_score:+.2f}")
print(f"CONFIDENCE: {confidence}")
print(f"WIDENING_EVIDENCE: {widening_evidence}")
print(f"TIGHTENING_EVIDENCE: {tightening_evidence}")
```

---
**Example output:**
```text

DRIVER_SIGNAL: widening
COMBINED_SCORE: +1.12
CONFIDENCE: medium
WIDENING_EVIDENCE: ['hyoas_observed_change_1b_bps_l1b', 'vix_log_ret_1b_l1b']
TIGHTENING_EVIDENCE: ['nasdaq_log_ret_1b_l1b']
```
Use this output to explain whether recent drivers support or challenge theforecast direction. 
Describe relationships as evidence or association, notproof of causality. 
Mention important context-only series separately when theyhelp explain the market regime.

## Notes on Gemini code execution limits

- Session timeout: ~30 seconds of CPU time. Keep computations lightweight.
- All data comes from the JSON payload; do not import repository constants.
- `import io` is available for parsing CSV strings.
- Do not attempt to `pip install` additional packages — the environment is fixed.
- Use `print()` to inspect intermediate results; the output is returned to you.
- `covariate_data_dictionary` is the source of truth for units and meaning.
