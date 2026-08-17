"""Run the CPO agent over a spec and score it beside the numerical baselines.

The agent is a :class:`Predictor` like any other, so this reuses the whole
:mod:`cpo.baselines` machinery -- same spec, same origins, same CRPS views.
The only thing that differs is the cost: one LLM call chain per origin, plus up
to three search-then-verify rounds inside each when news is on.  Start on the
two-origin smoke spec.

Credentials: ``OPENAI_BASE_URL`` and ``OPENAI_API_KEY`` (Vector proxy -- see
``planning-docs/vector-llm-proxy.md``), or failing that ``LLM_API_KEY`` with a
direct Gemini API key.  On the direct route the analyst defaults to
:data:`DIRECT_GEMINI_MODEL` and the web-news arm is unavailable (its
``search_web`` tool only exists behind the proxy).  This script checks up
front rather than letting the failure surface as an opaque ADK error.

Usage::

    PYTHONPATH=implementations uv run python implementations/cpo/run_agent.py            # smoke, news agent
    PYTHONPATH=implementations uv run python implementations/cpo/run_agent.py --arm basic
    PYTHONPATH=implementations uv run python implementations/cpo/run_agent.py --arm local  # news from the CSV
    PYTHONPATH=implementations uv run python implementations/cpo/run_agent.py --spec cutoffs --arm all
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from aieng.forecasting.evaluation.artifacts import cached_backtest
from cpo.agent import (
    CpoLocalNewsPromptBuilder,
    build_cpo_agent_predictor,
    build_cpo_basic_config,
    build_cpo_local_news_config,
    build_cpo_news_config,
)
from cpo.baselines import (
    _PACKAGE_DIR,
    attach_actuals,
    build_predictor,
    coverage,
    load_spec,
    mpob_service,
    per_origin,
    predictions_frame,
    run_predictor,
    skill_scores,
    summarise,
)
from dotenv import load_dotenv


SPECS = {name: _PACKAGE_DIR / "specs" / f"cpo_{name}.yaml" for name in ("smoke", "cutoffs", "backtest", "eval")}

#: Analyst model when running on the direct Gemini key rather than the proxy.
#: The proxy's default stays :data:`~aieng.forecasting.models.LITE_MODEL`;
#: ``--model`` overrides either.
DIRECT_GEMINI_MODEL = "gemini-3.1-pro-preview"

#: Artefact store, inside the implementation rather than at the repo-root cache
#: -- these results are committed (see .gitignore), because re-running an agent
#: arm costs tokens and the scored set is what every figure and table is built
#: from.
STORE_DIR = _PACKAGE_DIR / "data" / "predictions"

#: Baselines to score alongside the agent: the floor everything is judged
#: against, and the two strongest numerical models from the seven.
COMPARISON_BASELINES = ("naive", "ets", "autoarima")

REQUIRED_ENV = ("OPENAI_BASE_URL", "OPENAI_API_KEY")


def check_credentials() -> str:
    """Resolve which route the agent will take to a model, or exit.

    Preferred: the Vector proxy (``OPENAI_BASE_URL`` + ``OPENAI_API_KEY``),
    which every arm supports.  Fallback: a direct Gemini API key in
    ``LLM_API_KEY`` -- exported as ``GOOGLE_API_KEY``/``GEMINI_API_KEY`` so
    ADK's direct-provider routing picks it up.  Without either, ``AgentConfig``
    leaves both proxy fields ``None``, the ``LiteLlm`` wrap is skipped, and the
    failure surfaces as an opaque ADK error far from its cause -- hence the
    up-front check.

    Returns
    -------
    str
        ``"proxy"`` or ``"direct"``.
    """
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    if all(os.getenv(var) for var in REQUIRED_ENV):
        return "proxy"
    llm_key = os.getenv("LLM_API_KEY")
    if llm_key:
        # google-genai reads either name depending on version; set both.
        os.environ.setdefault("GOOGLE_API_KEY", llm_key)
        os.environ.setdefault("GEMINI_API_KEY", llm_key)
        return "direct"
    sys.exit(
        f"missing {', '.join(REQUIRED_ENV)} (and no LLM_API_KEY fallback) -- the agent cannot reach a model.\n"
        "Set the proxy vars in the repo-root .env (see planning-docs/vector-llm-proxy.md),\n"
        "set LLM_API_KEY to a Gemini API key, or run the bootcamp onboarding command from README.md."
    )


def main(argv: list[str] | None = None) -> None:
    """Run the requested agent arm(s) and print the comparison views."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", choices=sorted(SPECS), default="smoke", help="Which backtest spec to run.")
    parser.add_argument(
        "--arm",
        choices=("news", "basic", "local", "both", "all"),
        default="news",
        help=(
            "news = price + search_web; basic = price only; local = price + the "
            "committed weekly news CSV; both = basic and news; all = all three."
        ),
    )
    parser.add_argument("--model", default=None, help="Override the analyst model (default: the lite model).")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-run the agent even if a cached result exists (agent calls cost money -- off by default).",
    )
    parser.add_argument(
        "--baselines",
        nargs="*",
        default=list(COMPARISON_BASELINES),
        metavar="NAME",
        help="Numerical baselines to score alongside. Pass none to skip.",
    )
    args = parser.parse_args(argv)

    route = check_credentials()
    if route == "direct" and not args.model:
        args.model = DIRECT_GEMINI_MODEL

    chosen = {"both": ["basic", "news"], "all": ["basic", "news", "local"]}.get(args.arm, [args.arm])
    if route == "direct" and "news" in chosen:
        # search_web is a litellm call against the proxy base URL; there is no
        # direct-Gemini path for it, so the web-news arm cannot run here.
        if chosen == ["news"]:
            sys.exit("the news arm needs the Vector proxy (search_web has no direct-Gemini path); set OPENAI_BASE_URL.")
        chosen.remove("news")
        print("! skipping the news arm: search_web needs the Vector proxy, and this run is on the direct Gemini key")

    spec, svc = load_spec(SPECS[args.spec]), mpob_service()
    model_kw = {"model": args.model} if args.model else {}
    arms = {
        "basic": lambda: build_cpo_agent_predictor(build_cpo_basic_config(**model_kw)),
        "news": lambda: build_cpo_agent_predictor(build_cpo_news_config(**model_kw)),
        "local": lambda: build_cpo_agent_predictor(
            build_cpo_local_news_config(**model_kw), prompt_builder=CpoLocalNewsPromptBuilder()
        ),
    }

    frames = []
    for arm in chosen:
        print(f"running the {arm} agent over {len(spec.origin_dates or [])} origin(s)...", flush=True)
        # Cached by (spec_id, predictor_id): a re-run to rebuild a figure or a
        # table is free, and only --refresh spends tokens again.  The predictor
        # id folds in the model name, so swapping models cannot collide.
        result = cached_backtest(
            arms[arm](),
            spec,
            # BacktestSpec drops the yaml's spec_id field, so key the store on
            # the spec file's name -- same string, and it stays stable.
            spec_id=SPECS[args.spec].stem,
            data_service=svc,
            store_dir=STORE_DIR,
            force_refresh=args.refresh,
        )
        frames.append(predictions_frame(result))
        print(f"  {arm:6s} {result.predictor_id:48s} mean CRPS {result.mean_score:8.2f}")

    for name in args.baselines:
        result = run_predictor(build_predictor(name), spec, svc)
        frames.append(predictions_frame(result))
        print(f"  {name:6s} {result.predictor_id:48s} mean CRPS {result.mean_score:8.2f}")

    frame = attach_actuals(pd.concat(frames, ignore_index=True), svc)
    views = summarise(frame)
    print("\n=== mean CRPS by horizon (lower is better) ===")
    print(views["by_horizon"].to_string())
    print("\n=== mean CRPS at each cutoff ===")
    print(per_origin(frame).to_string())
    if "last_value_naive" in set(frame.predictor):
        print("\n=== skill vs naive (positive = beats 'nothing changes') ===")
        print(skill_scores(frame).to_string())
    cov = coverage(frame)
    print(f"\n=== q10-q90 coverage (nominal {cov.attrs['nominal']:.0%}) ===")
    print(cov.to_string())


if __name__ == "__main__":
    main()
