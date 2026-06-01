"""Skill state snapshot helpers for the adaptive agent training notebooks.

These utilities let NB05 and NB06 safely snapshot the strategy state before
a training or evaluation run, and restore it afterward if needed.

Snapshot contract
-----------------
Before any activity that mutates a strategy dir, call ``snapshot_state()``.
It copies ``skill_state.yaml`` → ``skill_state_pretrain.yaml`` inside the
same strategy directory.  If a snapshot already exists, the call is a no-op
(safe to re-run).

To undo all mutations (e.g. before repeating a training run from scratch),
call ``restore_state()``.  It copies the pretrain snapshot back over
``skill_state.yaml`` and re-renders ``SKILL.md`` via ``AdaptiveSkillStore``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from aieng.forecasting.methods.agentic.adaptive_skill import AdaptiveSkillStore
from energy_oil_forecasting.adaptive_agent.skill_state import WtiStrategyState


_YAML_FILENAME = "skill_state.yaml"
_PRETRAIN_FILENAME = "skill_state_pretrain.yaml"


def snapshot_state(strategy_dir: Path, *, overwrite: bool = False) -> Path:
    """Copy ``skill_state.yaml`` to ``skill_state_pretrain.yaml``.

    Parameters
    ----------
    strategy_dir : Path
        The strategy skill directory (e.g. ``skills/wti-strategy-stats``).
    overwrite : bool, default=False
        If ``False`` and a snapshot already exists, the call is a no-op.
        Set ``True`` to force-overwrite an existing snapshot.

    Returns
    -------
    Path
        Path to the snapshot file.
    """
    src = strategy_dir / _YAML_FILENAME
    dst = strategy_dir / _PRETRAIN_FILENAME
    if dst.exists() and not overwrite:
        print(f"  [snapshot_state] Snapshot already exists: {dst.name} — skipping.")
        return dst
    shutil.copy2(src, dst)
    print(f"  [snapshot_state] Saved → {dst.relative_to(strategy_dir.parent.parent)}")
    return dst


def restore_state(strategy_dir: Path) -> None:
    """Restore ``skill_state.yaml`` from the pre-training snapshot and re-render ``SKILL.md``.

    Parameters
    ----------
    strategy_dir : Path
        The strategy skill directory to restore.

    Raises
    ------
    FileNotFoundError
        If no pretrain snapshot exists (``snapshot_state`` was never called).
    """
    src = strategy_dir / _PRETRAIN_FILENAME
    if not src.exists():
        raise FileNotFoundError(
            f"No pretrain snapshot found at {src}. Call snapshot_state() before running training activities."
        )
    dst = strategy_dir / _YAML_FILENAME
    shutil.copy2(src, dst)
    # Re-render SKILL.md from the restored YAML
    store: AdaptiveSkillStore[WtiStrategyState] = AdaptiveSkillStore(
        skill_dir=strategy_dir,
        state_type=WtiStrategyState,
    )
    state = store.load()
    store.save(state)
    print(f"  [restore_state] Restored {strategy_dir.name} from pretrain snapshot.")


def state_checksum(strategy_dir: Path) -> str:
    """Return a content hash of ``skill_state.yaml`` for before/after comparison.

    Used in NB06 to verify the agent's state was not mutated during evaluation.

    Parameters
    ----------
    strategy_dir : Path
        The strategy skill directory.

    Returns
    -------
    str
        Hex digest of the YAML content.
    """
    import hashlib  # noqa: PLC0415

    content = (strategy_dir / _YAML_FILENAME).read_bytes()
    return hashlib.sha256(content).hexdigest()
