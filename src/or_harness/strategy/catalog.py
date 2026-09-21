"""Built-in strategy catalog: data-driven, loaded from catalog.json.

The catalog carries structural vocabulary ONLY (applicability, actions,
fallbacks, solver family) — deliberately NO quality/cost priors, so a cold
start honestly reports ``no_memory``. Quality/cost expectations come from
evidence (entries and conditional statistics), never from the catalog; the
record/induction loop builds them from real executions.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Dict, List, Optional

from or_harness.core.schema import Strategy


def load_catalog(path: Optional[str] = None) -> Dict[str, Strategy]:
    """Load the catalog as {strategy_id: Strategy}. Deterministic order."""
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    else:
        with resources.files("or_harness.strategy").joinpath("catalog.json").open(
                "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    strategies: Dict[str, Strategy] = {}
    for item in raw["strategies"]:
        strategy = Strategy.from_dict(item)
        if strategy.strategy_id in strategies:
            raise ValueError(f"duplicate strategy_id {strategy.strategy_id!r}")
        strategies[strategy.strategy_id] = strategy
    return strategies

