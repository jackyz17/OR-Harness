"""Built-in strategy catalog: data-driven, loaded from catalog.json.

Priors in the catalog are the selector's only evidence at cold start. They are
deliberately approximate; the record/induction loop corrects them from real
executions (C2 criterion: systematic prior divergence is itself an induction
trigger).
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


def strategy_list(catalog: Dict[str, Strategy]) -> List[Strategy]:
    return [catalog[k] for k in sorted(catalog)]
