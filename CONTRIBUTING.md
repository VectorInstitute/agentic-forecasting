# Contributing to aieng-template

Thanks for your interest in contributing to the aieng-template-implementation!

To submit PRs, please fill out the PR template along with the PR. If the PR
fixes an issue, don't forget to link the PR to the issue!

## Code checks (before opening a PR)

Commits do **not** run linters automatically. After `uv sync --group dev`, run the same checks CI uses:

```bash
uv run pre-commit run --all-files
```

Or use **`make dev lint`** to sync the dev environment, apply Ruff formatting/import sorting, and run mypy on the `aieng` package. Fix any failures before submitting a PR — GitHub Actions runs `pre-commit run --all-files` on pushes and pull requests to `main`.

## Coding guidelines

For code style, we recommend the [PEP 8 style guide](https://peps.python.org/pep-0008/).

For docstrings we use [numpy format](https://numpydoc.readthedocs.io/en/latest/format.html).

We use [ruff](https://docs.astral.sh/ruff/) for code formatting and static code
analysis. Ruff checks various rules including [flake8](https://docs.astral.sh/ruff/faq/#how-does-ruff-compare-to-flake8). Address ruff and mypy feedback from your local run or from CI before merging.

Last but not the least, we use type hints in our code which is then checked using
[mypy](https://mypy.readthedocs.io/en/stable/).
