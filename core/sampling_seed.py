"""Deterministic per-run sampling seeds, so two benchmark runs are comparable.

Stage1 samples at temperature 0.5 and three runs per agent, which makes the
exact-match total move on its own: level1_final_06, _07 and _08 all scored 19 of
53 while 8 tasks flipped between consecutive runs. A change worth one or two
tasks cannot be read against that, and three rounds of fixes were measured only
through intermediate metrics for exactly this reason.

SCP_STAGE1_SEED pins the sampling so a rerun of unchanged code reproduces its
result, which is what makes a small change legible. The seed still varies per
agent and per run -- the three runs have to keep diverging or their aggregation
means nothing -- it just varies the same way every time.

It is pinned by default. Free sampling is still available, but it has to be
asked for (`SCP_STAGE1_SEED=off`, or `--stage1-seed off`) rather than being
what you get by forgetting, because an unpinned run cannot be compared against
anything.
"""

from __future__ import annotations

import hashlib
import os

ENV_VAR = "SCP_STAGE1_SEED"
DEFAULT_SEED = 42
_FREE_VALUES = frozenset({"off", "none", "free", "random"})
_MAX_SEED = 2**31 - 1


def base_seed() -> int | None:
    """Read the configured base seed, or None when sampling is set free.

    Raises on an unreadable value rather than falling back: silently unpinning
    would produce a run that looks comparable and is not, which is the failure
    this module exists to prevent.
    """

    raw = str(os.getenv(ENV_VAR, "") or "").strip()
    if not raw:
        return DEFAULT_SEED
    if raw.casefold() in _FREE_VALUES:
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"{ENV_VAR}={raw!r} is neither an integer nor one of "
            f"{sorted(_FREE_VALUES)}"
        ) from None


def describe() -> str:
    """One line for run logs and reports, so an output folder says how it ran."""

    seed = base_seed()
    return "free (unpinned)" if seed is None else str(seed)


def run_seed(*, agent_id: str, run_index: int, turn: int = 0) -> int | None:
    """Derive this run's seed, or None when sampling is set free.

    Derived with blake2s rather than hash() because str hashing is salted per
    process, which would make the seed differ between runs of the same code and
    defeat the point.
    """

    base = base_seed()
    if base is None:
        return None
    digest = hashlib.blake2s(
        f"{base}:{agent_id}:{run_index}:{turn}".encode("utf-8"),
        digest_size=4,
    ).digest()
    return int.from_bytes(digest, "big") % _MAX_SEED


def sampling_overrides(
    *,
    agent_id: str,
    run_index: int,
    turn: int = 0,
) -> dict[str, int]:
    """Completion overrides to pin sampling, empty when it stays free."""

    seed = run_seed(agent_id=agent_id, run_index=run_index, turn=turn)
    return {} if seed is None else {"seed": seed}


__all__ = [
    "DEFAULT_SEED",
    "ENV_VAR",
    "base_seed",
    "describe",
    "run_seed",
    "sampling_overrides",
]
