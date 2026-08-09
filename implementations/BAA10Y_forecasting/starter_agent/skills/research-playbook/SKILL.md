---
name: research-playbook
description: >-
  How to use the search_web tool well when grounding a forecast in recent
  news — phrase cutoff-aware queries, decide what is worth searching for, and
  weigh sources. Load this before your first search_web call. No scripts.
---

# Research playbook

A short guide to getting real signal out of `search_web`. This is a starter
skill — extend it with the queries and sources that work for your problem.

## The one rule that matters

Always pass `cutoff_date` equal to the `as_of` date in your payload. It is the
temporal fence that keeps post-origin information out of a historical forecast.
A forecast that "knew" what happened after `as_of` is not a forecast.

`search_web` runs an independent verifier on every result and returns
`[SEARCH_VERIFICATION_FAILED]` instead of content it couldn't confirm as
pre-cutoff. Treat that as no verified news for the query — proceed on your
other signals and say so, never filling the gap from your own background
knowledge.

## How to search

- **Search before you forecast, not after.** Gather context first, then reason.
- **One topic per query.** Several focused queries beat one broad one. Stop when
  new queries stop returning new facts.
- **Ask for the present state, not a prediction.** "current OPEC+ production
  policy" returns facts; "will oil go up" returns noise.
- **Weigh sources.** Prefer primary releases and major outlets; treat a single
  blog or forum post as a lead to confirm, not a fact.

## Domain focus (edit this for your use case)

For BAA10Y spread changes, focus searches on the current state of:

- Federal Reserve policy and interest-rate guidance;
- inflation and employment releases;
- recession and economic-growth indicators;
- corporate defaults, downgrades, and credit-rating developments;
- corporate earnings, refinancing pressure, and funding conditions;
- financial-sector stress and market liquidity;
- VIX, equity-market stress, and investor risk sentiment;
- major geopolitical or policy shocks affecting credit markets.

Search for information available on or before the forecast cutoff. Give greater
weight to Federal Reserve releases, official economic data, rating agencies,
company disclosures, and established financial-news sources.

Use news primarily to identify the current credit-risk regime and possible tail
risk. Keep the directional forecast conservative unless multiple verified
signals consistently support spread widening or tightening.

## Room to grow

- Add a curated list of go-to sources for your domain.
- Track which queries paid off and prune the ones that didn't.
- Add a `references/` file with example high-signal searches.
