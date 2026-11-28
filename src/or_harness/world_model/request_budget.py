"""Provider-request budget: measured size, independent block ceilings, trimming.

The prediction request is assembled from a system prompt, the candidate, the
frozen context and several evidence blocks. Three things must be true for the
budget to be trustworthy:

- the WHOLE request is measured (system prompt + JSON body), not just the
  context fragment;
- a real token count is used when the provider exposes one, otherwise the
  count is an explicitly labelled CONSERVATIVE estimate (CJK and dense JSON
  text cost more tokens per character than plain Western prose, so the old
  default ``chars / 4`` under-counted them);
- when the request is over budget, the HISTORY FEEDBACK is reduced first
  (per-row before per-case), then other non-core/evidence blocks — the
  current task, the candidate, the key execution constraints and the
  prediction contract are never silently truncated. If the protected core
  alone exceeds the budget the request is NOT sent and the failure is
  reported.

Everything here is a VIEW concern: no stored evidence, no calibration and
no original prediction is modified.
"""
from __future__ import annotations

import copy
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from or_harness.strategy.evidence_views import (
    PROVIDER_FEEDBACK_VERSION,
    gcr_evidence_budget_chars,
    history_feedback_budget_chars,
    noncore_budget_chars,
    provider_paired_feedback,
)

#: Calibrated for a conservative OVER-estimate of tokens, never an exact
#: tokenizer: a tokenizer count is used when the provider exposes one.
ASCII_CHARS_PER_TOKEN = 3.0
#: CJK text is roughly one token per character (often more), not four
#: characters per token: the old default badly under-counted it.
CJK_TOKENS_PER_CHAR = 1.2
NON_ASCII_TOKENS_PER_CHAR = 1.0

#: Blocks whose size may be reduced (each has its own ceiling). The core
#: fields below are NOT in this list: trimming them would answer a different
#: question or make the answer meaningless.
PROTECTED_CONTEXT_KEYS = (
    "joint_problem",
    "execution_constraints",
    "solving_context",
    "strategy_outcome_calibration",
    "task_id",
    "task_digest",
    "effective_input_digest",
    "snapshot_id",
    "context_id",
    "context_version",
    "episode_id",
)
PROTECTED_REQUEST_KEYS = (
    "risk_convention",
    "benefit_convention",
    "cost_scope",
    "output_contract",
    "hplus_convention",
    "prediction_context",
    "candidate",
    "request_kind",
)

#: Keys that are LOCAL diagnostics, never model input. The provider receives
#: a request WITHOUT them, so the budget must measure the same object that is
#: actually serialized and sent (``request_size`` describes that sent object;
#: including it in the measured body would understate the real payload).
WIRE_EXCLUDED_REQUEST_KEYS = ("request_size", "request_diagnostics")


def strip_request_diagnostics(request: Dict[str, Any]) -> Dict[str, Any]:
    """The request AS SENT: a shallow copy without local-only diagnostics."""
    return {key: value for key, value in (request or {}).items()
            if key not in WIRE_EXCLUDED_REQUEST_KEYS}


#: Trimming order for the CONTEXT fragment: history first, then other
#: non-core material, then the calibration/evidence blocks. The current task
#: body, its constraints and the calibration selection are never in this
#: list.
TRIMMABLE_CONTEXT_BLOCKS = (
    "prediction_execution_pairs",
    "harness_capability",
    "capability_version",
    "retrieval_evidence",
    "hplus_feedback",
    "prediction_reminders",
    "cost_evidence",
    "implementation_r_evidence",
    "notes",
)


def _size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                          default=str))


def _is_cjk(code_point: int) -> bool:
    return (
        0x4E00 <= code_point <= 0x9FFF      # CJK unified ideographs
        or 0x3400 <= code_point <= 0x4DBF   # extension A
        or 0x20000 <= code_point <= 0x2FA1F  # extension B-F
        or 0x3000 <= code_point <= 0x303F   # CJK punctuation
        or 0x3040 <= code_point <= 0x30FF   # hiragana / katakana
        or 0xAC00 <= code_point <= 0xD7AF   # hangul syllables
        or 0xF900 <= code_point <= 0xFAFF   # compatibility ideographs
    )


def estimate_tokens(text: str) -> Dict[str, Any]:
    """A LABELLED conservative token estimate for one text.

    ``label`` is always ``labeled_conservative_estimate``; a caller with a
    real tokenizer must not use this value as a token count.
    """
    text = str(text or "")
    ascii_chars = 0
    cjk_chars = 0
    other_chars = 0
    for char in text:
        code_point = ord(char)
        if code_point < 128:
            ascii_chars += 1
        elif _is_cjk(code_point):
            cjk_chars += 1
        else:
            other_chars += 1
    tokens = (ascii_chars / ASCII_CHARS_PER_TOKEN
              + cjk_chars * CJK_TOKENS_PER_CHAR
              + other_chars * NON_ASCII_TOKENS_PER_CHAR)
    return {
        "tokens": int(math.ceil(tokens)),
        "basis": "labeled_conservative_estimate",
        "estimator": (
            f"tokens = ascii/{ASCII_CHARS_PER_TOKEN:g} + "
            f"cjk*{CJK_TOKENS_PER_CHAR:g} + other*"
            f"{NON_ASCII_TOKENS_PER_CHAR:g}, rounded up; a CONSERVATIVE "
            "ESTIMATE, not a tokenizer count"),
        "chars": len(text),
        "ascii_chars": ascii_chars,
        "cjk_chars": cjk_chars,
        "other_non_ascii_chars": other_chars,
    }


def _provider_token_count(provider: Any, text: str) -> Optional[int]:
    """Use the provider's own tokenizer when it exposes one."""
    counter = getattr(provider, "count_tokens", None)
    if not callable(counter):
        return None
    try:
        value = counter(text)
    except TypeError:
        try:
            value = counter(text=text)
        except Exception:
            return None
    except Exception:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0 or math.isnan(float(value)) or math.isinf(float(value)):
        return None
    return int(math.ceil(float(value)))


def measure_text(text: str, *, provider: Any = None) -> Dict[str, Any]:
    """Measure one piece of text: real tokenizer when available, else estimate."""
    real = _provider_token_count(provider, text)
    if real is not None:
        return {"tokens": real, "basis": "provider_tokenizer",
                "estimator": ("the provider's own count_tokens(); a real "
                              "token count, not an estimate"),
                "chars": len(text)}
    return estimate_tokens(text)


def measure_request(request: Dict[str, Any], system_prompt: str = "",
                    *, provider: Any = None) -> Dict[str, Any]:
    """Measure the WHOLE request AS SENT (system prompt + wire JSON body).

    Local diagnostics (``request_size``) are excluded because the provider
    never receives them; the budget therefore describes exactly the object
    that is serialized on the wire. The serialization call is the same one
    the HTTP provider uses (``json.dumps(..., ensure_ascii=False,
    default=str)``).
    """
    body = json.dumps(strip_request_diagnostics(request),
                      ensure_ascii=False, default=str)
    prompt = str(system_prompt or "")
    body_measure = measure_text(body, provider=provider)
    prompt_measure = measure_text(prompt, provider=provider)
    measured = {
        "tokens": int(body_measure["tokens"]) + int(prompt_measure["tokens"]),
        "basis": (body_measure["basis"] if body_measure["basis"]
                  == prompt_measure["basis"] else
                  "provider_tokenizer+labeled_estimate"),
        "estimator": (
            "sum of separately measured system prompt and JSON body; "
            + ("a real tokenizer count" if body_measure["basis"]
               == "provider_tokenizer" else
               "a LABELLED CONSERVATIVE ESTIMATE (ascii/3 + cjk*1.2 + "
               "other, rounded up), never presented as a tokenizer count")),
        "body_chars": len(body),
        "system_prompt_chars": len(prompt),
        "chars": len(body) + len(prompt),
    }
    if body_measure["basis"] == prompt_measure["basis"] == \
            "labeled_conservative_estimate":
        measured["cjk_chars"] = (body_measure.get("cjk_chars", 0)
                                 + prompt_measure.get("cjk_chars", 0))
        measured["ascii_chars"] = (body_measure.get("ascii_chars", 0)
                                   + prompt_measure.get("ascii_chars", 0))
        measured["estimate_label"] = (
            "labeled_conservative_estimate: CJK/JSON-dense text is counted "
            "more heavily than the old chars/4 default")
    return measured


def input_token_budget() -> Dict[str, Any]:
    """The deployment's input token budget (shared with the context bound)."""
    from or_harness.world_model.context import (  # lazy: avoid a cycle
        _input_budget_chars,
    )
    chars, basis = _input_budget_chars()
    return {
        "tokens": int(basis["input_tokens"]),
        "declared_tokens": int(basis["budget_tokens"]),
        "reserved_output_tokens": int(basis["reserved_output_tokens"]),
        "chars_per_token": basis.get("chars_per_token"),
        "char_bound": chars,
        "source": basis.get("source"),
    }


# ---------------------------------------------------------------------------
# deterministic block reduction
# ---------------------------------------------------------------------------

def _history_positions(block: Dict[str, Any]) -> List[str]:
    refs = []
    for pair in block.get("pairs") or []:
        if not isinstance(pair, dict):
            continue
        ref = pair.get("detail_ref") or pair.get("evaluation_id")
        if ref:
            refs.append(str(ref))
    return refs


def trim_history_block(block: Dict[str, Any], max_chars: int, *,
                       stage: str = "request") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Reduce ONE provider history block to ``max_chars``, per-row first.

    The block is already ordered by value (labeled cases first, execution
    errors last, then relevance), so rows are dropped from the TAIL. Every
    dropped row's reference is reported; when even the counts-only summary
    is too large the block becomes an explicit omission report (never a
    silent empty history).
    """
    block = copy.deepcopy(block or {})
    before = _size(block)
    info: Dict[str, Any] = {
        "stage": stage,
        "before_chars": before,
        "dropped_pairs": 0,
        "dropped_refs": [],
        "dropped_comparisons": 0,
        "summarised": False,
    }
    pairs = block.get("pairs") if isinstance(block.get("pairs"), list) else []
    while _size(block) > max_chars and pairs:
        row = pairs.pop()
        if isinstance(row, dict):
            ref = row.get("detail_ref") or row.get("evaluation_id")
            if ref and len(info["dropped_refs"]) < 20:
                info["dropped_refs"].append(str(ref))
        info["dropped_pairs"] += 1
        block["pairs"] = pairs
        available = block.get("n_pairs_available")
        if available is not None:
            block["n_pairs_omitted"] = max(
                0, int(available) - len(pairs))
        else:
            block["n_pairs_omitted"] = int(
                block.get("n_pairs_omitted") or 0) + 1
        block["n_pairs_included"] = len(pairs)
        block["n_pairs_provider_dropped"] = int(
            block.get("n_pairs_provider_dropped") or 0) + 1
    if _size(block) > max_chars and block.get("same_task_comparisons"):
        info["dropped_comparisons"] = len(block["same_task_comparisons"])
        block["same_task_comparisons"] = []
        block["same_task_comparisons_note"] = (
            "side-by-side comparisons were omitted to keep the request "
            "within its size bound; the pair rows and `orx calibration` "
            "retain the facts")
    if _size(block) > max_chars and not pairs:
        summary = {
            "omitted_for_size": True,
            "feedback_version": block.get("feedback_version"),
            "provider_view_version": block.get("provider_view_version"),
            "n_pairs_total": block.get("n_pairs_total"),
            "n_pairs_available": block.get("n_pairs_available"),
            "n_pairs_included": 0,
            "n_pairs_omitted": block.get("n_pairs_omitted"),
            "budget_chars": max_chars,
            "detail_ref": "orx calibration",
            "note": ("the provider-only history block was reduced to counts "
                     "to keep the request within its size bound; every "
                     "dropped case and field remains in the stored block, "
                     "`orx calibration` and the evaluation store"),
        }
        block = summary
        info["summarised"] = True
        info["dropped_pairs"] = len(pairs) if pairs else info["dropped_pairs"]
    if info["dropped_pairs"]:
        block.setdefault("omission_note", (
            f"{info['dropped_pairs']} pair(s) were dropped to keep the "
            "request within its size bound; the stored block and "
            "`orx calibration` retain them"))
        if info["dropped_refs"]:
            block["dropped_pair_refs"] = info["dropped_refs"]
    info["after_chars"] = _size(block)
    return block, info


def _cap_list(block: Dict[str, Any], path: Tuple[str, ...], limit: int,
              removed: List[Dict[str, Any]]) -> None:
    node: Any = block
    for key in path[:-1]:
        if not isinstance(node, dict) or key not in node:
            return
        node = node[key]
    key = path[-1]
    if not isinstance(node, dict):
        return
    value = node.get(key)
    if isinstance(value, list) and len(value) > limit:
        removed.append({"field": ".".join(path),
                        "n_removed": len(value) - limit})
        node[key] = value[:limit]


def trim_evidence_block(block: Dict[str, Any], max_chars: int, *,
                        kind: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Reduce one G/C/R evidence block ceiling, keeping the applicable part.

    The selection inside each block is already relevance-ordered, so dropping
    tail list entries is deterministic. Counts/notes record what was removed
    and ``detail_ref`` points at the full population in the stored context.
    """
    block = copy.deepcopy(block or {})
    before = _size(block)
    removed: List[Dict[str, Any]] = []
    detail_ref = {
        "implementation_r_evidence": (
            "orx calibration #technical_r / stored prediction context"),
        "cost_evidence": "orx calibration #conditional_cost / stored context",
        "retrieval_evidence": "orx inspect --bank experience / stored context",
        "capability": "stored prediction context",
    }.get(kind, "stored prediction context")
    if kind == "g_candidate_evidence":
        from or_harness.world_model.history_retrieval import (
            refresh_g_history_coverage,
        )
        refresh_g_history_coverage(block)
    elif kind == "implementation_r_evidence":
        from or_harness.world_model.history_retrieval import (
            refresh_r_history_coverage,
        )
        node = block
        if "event_definitions" in node:
            node.pop("event_definitions", None)
            removed.append({"field": "event_definitions",
                            "ref": "request.risk_convention.event_definitions"})
        # The GLOBAL base rates are scalar counts: they are background the
        # model must keep even when local case bodies are reduced.
        for entry in (block.get("cases") or []):
            if isinstance(entry, dict):
                for key in ("source_refs",):
                    values = entry.get(key)
                    if isinstance(values, list) and len(values) > 3:
                        removed.append({"field": key,
                                        "n_removed": len(values) - 3})
                        entry[key] = values[:3]
        refresh_r_history_coverage(block)
    elif kind == "cost_evidence":
        from or_harness.world_model.history_retrieval import (
            refresh_c_history_coverage,
        )
        conditions = block.get("current_conditions")
        if isinstance(conditions, dict):
            _cap_list(conditions, ("reusable_artifacts",), 2, removed)
        refresh_c_history_coverage(block)
    elif kind == "retrieval_evidence":
        hits = block.get("hits")
        if isinstance(hits, list) and len(hits) > 8:
            removed.append({"field": "hits", "n_removed": len(hits) - 8})
            block["hits"] = hits[:8]
        for key in ("recommendations", "vector_recall"):
            value = block.get(key)
            if isinstance(value, list) and len(value) > 5:
                removed.append({"field": key, "n_removed": len(value) - 5})
                block[key] = value[:5]
    elif kind == "capability":
        from or_harness.world_model.context import _compact_capability
        if "sources" in block or "notes" in block or "evidence" in block:
            compact = _compact_capability(block)
            compact["budget_trimmed"] = True
            block = compact
            removed.append({"field": "per-source detail",
                            "ref": detail_ref})
    # Generic tail reduction when the specific edits were not enough.
    while _size(block) > max_chars:
        progress = False
        for key, value in list(block.items()):
            if isinstance(value, list) and len(value) > 0:
                block[key] = value[:-1]
                removed.append({"field": key, "n_removed": 1})
                progress = True
            elif isinstance(value, dict) and key not in ("note",):
                for inner, inner_value in list(value.items()):
                    if isinstance(inner_value, list) and inner_value:
                        value[inner] = inner_value[:-1]
                        removed.append(
                            {"field": f"{key}.{inner}", "n_removed": 1})
                        progress = True
                        break
            if progress:
                break
        if not progress:
            break
    if kind == "cost_evidence":
        from or_harness.world_model.history_retrieval import (
            refresh_c_history_coverage,
        )
        refresh_c_history_coverage(block)
    elif kind == "implementation_r_evidence":
        from or_harness.world_model.history_retrieval import (
            refresh_r_history_coverage,
        )
        refresh_r_history_coverage(block)
    elif kind == "g_candidate_evidence":
        from or_harness.world_model.history_retrieval import (
            refresh_g_history_coverage,
        )
        refresh_g_history_coverage(block)
    if removed:
        block["budget_trim"] = {
            "removed": removed,
            "detail_ref": detail_ref,
            "note": ("this block was reduced to its own provider budget; "
                     "the full population stays in the stored context"),
        }
    info = {"before_chars": before, "after_chars": _size(block),
            "removed": removed}
    return block, info


def _fit_blocks(view: Dict[str, Any], keys: Tuple[str, ...], budget: int,
                *, kind_for) -> Dict[str, Any]:
    """Keep a set of blocks within one shared ceiling by shrinking the biggest."""
    report: Dict[str, Any] = {"budget_chars": budget, "blocks": {}}
    sizes = {key: _size(view[key]) for key in keys if key in view}
    report["blocks"] = dict(sizes)
    total = sum(sizes.values())
    report["before_chars"] = total
    rounds = 0
    while total > budget and sizes and rounds < 24:
        rounds += 1
        key = max(sizes, key=lambda item: sizes[item])
        block = view[key]
        target = max(300, int(sizes[key] * 0.6))
        if isinstance(block, dict):
            trimmed, info = trim_evidence_block(block, target,
                                                kind=kind_for(key))
        elif isinstance(block, list):
            trimmed = block[:max(0, int(len(block) * 0.6))]
            info = {"before_chars": sizes[key],
                    "after_chars": _size(trimmed)}
        else:
            break
        view[key] = trimmed
        sizes[key] = _size(trimmed)
        total = sum(sizes.values())
        report["blocks"][key] = sizes[key]
    report["after_chars"] = total
    report["trimmed"] = total != report["before_chars"]
    report["rounds"] = rounds
    report["detail_ref"] = "the stored prediction context"
    return report


def apply_provider_block_budgets(view: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the three INDEPENDENT provider-only ceilings to a context view.

    - history feedback: its own character ceiling, reduced per-row first;
    - G/C/R evidence (cost + technical R): one shared ceiling;
    - other non-core material (capability, retrieval, H+ feedback, notes):
      one shared ceiling.

    The current task, its execution constraints and the calibration
    selection are NOT touched here.
    """
    limits: Dict[str, Any] = {}
    block = view.get("prediction_execution_pairs")
    if isinstance(block, dict) and block:
        budget = history_feedback_budget_chars()
        # The context already projects the block; projecting an ALREADY
        # projected block again would be a lossy second pass. Only project
        # a raw stored/selected block once.
        if block.get("provider_view_version") == PROVIDER_FEEDBACK_VERSION:
            projected = copy.deepcopy(block)
        else:
            projected = provider_paired_feedback(block)
        projected_chars = _size(projected)
        if projected_chars > budget:
            projected, info = trim_history_block(projected, budget,
                                                 stage="provider_view")
            info["budget_chars"] = budget
            info["chars"] = _size(projected)
            info["trimmed"] = True
            limits["history_feedback"] = info
        else:
            limits["history_feedback"] = {
                "budget_chars": budget,
                "chars": projected_chars,
                "trimmed": bool(projected.get("n_pairs_provider_dropped")),
            }
        view["prediction_execution_pairs"] = projected
    evidence_keys = tuple(key for key in ("cost_evidence",
                                         "implementation_r_evidence",
                                         "g_candidate_evidence")
                          if key in view)
    if evidence_keys:
        limits["gcr_evidence"] = _fit_blocks(
            view, evidence_keys, gcr_evidence_budget_chars(),
            kind_for=lambda key: key)
    # Retrieval evidence is EVIDENCE, not "other non-core material": it is
    # governed by the whole-request budget, never squeezed by the non-core
    # ceiling (which could otherwise empty the very rule body the agent must
    # read). It is reduced only when the WHOLE request is over budget.
    noncore_keys = tuple(key for key in (
        "harness_capability", "capability_version",
        "hplus_feedback", "prediction_reminders", "notes") if key in view)
    if noncore_keys:
        limits["noncore"] = _fit_blocks(
            view, noncore_keys, noncore_budget_chars(),
            kind_for=lambda key: ("capability" if key == "harness_capability"
                                  else "generic"))
    view["budget_limits"] = limits
    return view


# ---------------------------------------------------------------------------
# whole-request enforcement
# ---------------------------------------------------------------------------

def _core_sizes(request: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
    context = request.get("prediction_context")
    context = context if isinstance(context, dict) else {}
    core_context = {key: _size(context[key]) for key in PROTECTED_CONTEXT_KEYS
                    if key in context}
    core_request = {key: _size(request[key]) for key in PROTECTED_REQUEST_KEYS
                    if key in request and key != "prediction_context"}
    return {
        "system_prompt_chars": len(system_prompt or ""),
        "request_contract_chars": core_request,
        "context_core_chars": core_context,
        "core_chars": (len(system_prompt or "") + sum(core_request.values())
                       + sum(core_context.values())),
        "note": ("these fields are never truncated by the budget: the "
                 "current task, the candidate, the execution constraints and "
                 "the prediction contract"),
    }


def _history_summary(block: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "omitted_for_size": True,
        "feedback_version": block.get("feedback_version"),
        "provider_view_version": block.get("provider_view_version"),
        "n_pairs_total": block.get("n_pairs_total"),
        "n_pairs_included": 0,
        "n_pairs_omitted": block.get("n_pairs_omitted"),
        "detail_ref": "orx calibration",
        "note": ("the whole history block was reduced to its counts to keep "
                 "the request within its size bound; the stored block and "
                 "`orx calibration` retain every case"),
    }


def _minimal_global_g(global_history: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("n_observed", "n_predicted", "n_scored", "n_distinct_tasks",
            "n_distinct_episodes", "observed_mean", "predicted_mean", "mae",
            "signed_error_mean", "bias_direction", "brier", "auc", "support")
    return {key: global_history.get(key) for key in keys
            if global_history.get(key) is not None}


def _minimal_global_cost(global_cost: Dict[str, Any]) -> Dict[str, Any]:
    dimensions = {}
    for dim, entry in (global_cost.get("dimensions") or {}).items():
        if not isinstance(entry, dict):
            continue
        dimensions[dim] = {key: entry.get(key) for key in (
            "n", "p50", "p90", "n_predicted_actual", "error_direction",
            "n_incomplete") if entry.get(key) is not None}
    return {"dimensions": dimensions, "support": global_cost.get("support")}


def _minimal_global_r(block: Dict[str, Any]) -> Dict[str, Any]:
    events = {}
    for name, entry in (block.get("global_event_base_rates") or {}).items():
        if not isinstance(entry, dict):
            continue
        events[name] = {key: entry.get(key) for key in (
            "occurred", "not_occurred", "unknown", "n_known_labels",
            "occurrence_rate") if entry.get(key) is not None}
    performance = {}
    for name, entry in (block.get("global_prediction_performance")
                        or {}).items():
        if not isinstance(entry, dict):
            continue
        performance[name] = {key: entry.get(key) for key in (
            "n_scorable", "mean_brier", "brier_support")
            if entry.get(key) is not None}
    return {"events": events, "prediction_performance": performance}


def _gcr_evidence_summary(key, block):
    """The smallest honest form of a unified-history G/C/R evidence block.

    Used only when the WHOLE request is still over its bound after per-block
    trimming. It keeps the GRADES, the coverage counts, the MINIMAL global
    values (counts, distribution percentiles, error direction, event rates)
    and the source pointers — and drops the per-case bodies, which remain in
    the stored context.
    """
    block = block if isinstance(block, dict) else {}
    detail_ref = "stored prediction context #history_retrieval"
    if key == "cost_evidence":
        return {
            "version": block.get("version"),
            "observation_rule_version": block.get("observation_rule_version"),
            "current_conditions": copy.deepcopy(block.get("current_conditions")),
            "reference_coverage": copy.deepcopy(block.get("reference_coverage")),
            "evidence_status": block.get("evidence_status"),
            "cases": [],
            "global_cost": _minimal_global_cost(
                block.get("global_cost") or {}),
            "missing": list(block.get("missing") or []),
            "applicability": block.get("applicability"),
            "omitted_for_size": True,
            "detail_ref": detail_ref,
            "note": ("per-case cost bodies were omitted to keep the full "
                     "request within its bound; the strict/broader counts and "
                     "the GLOBAL cost distribution are retained and the "
                     "cases remain in the stored context"),
        }
    if key == "implementation_r_evidence":
        return {
            "version": block.get("version"),
            "solver_interface": block.get("solver_interface"),
            "event_definitions": copy.deepcopy(
                block.get("event_definitions") or {}),
            "selection": copy.deepcopy(block.get("selection")),
            "cases": [],
            **_minimal_global_r(block),
            "missing": list(block.get("missing") or []),
            "support": block.get("support"),
            "global_scope": block.get("global_scope"),
            "omitted_for_size": True,
            "detail_ref": detail_ref,
            "note": ("local technical cases were omitted to keep the full "
                     "request within its bound; the global event rates and "
                     "prediction performance remain, and the per-case facts "
                     "remain in the stored context"),
        }
    return {
        "version": block.get("version"),
        "selection": copy.deepcopy(block.get("selection")),
        "global_history": _minimal_global_g(
            block.get("global_history") or {}),
        "degraded": copy.deepcopy(block.get("degraded") or []),
        "cases": [],
        "omitted_for_size": True,
        "detail_ref": detail_ref,
        "note": ("local G cases were omitted to keep the full request within "
                 "its bound; the global G diagnostic remains and the full "
                 "cases remain in the stored context"),
    }


def enforce_full_request_budget(request: Dict[str, Any], system_prompt: str,
                                *, provider: Any = None) -> Dict[str, Any]:
    """Measure the WHOLE request and enforce the deployment's input bound.

    Returns the ``request_size`` block and mutates ``request`` only in the
    trim-able evidence blocks. ``sendable`` is false only when the protected
    core (system prompt + candidate + task + constraints + contract) alone
    exceeds the budget — an over-budget request is never silently sent.
    """
    budget = input_token_budget()
    max_tokens = int(budget["tokens"])
    before = measure_request(request, system_prompt, provider=provider)
    stages: List[Dict[str, Any]] = []

    def measured() -> Dict[str, Any]:
        return measure_request(request, system_prompt, provider=provider)

    current = before
    if current["tokens"] > max_tokens:
        context = request.get("prediction_context")
        context = context if isinstance(context, dict) else {}
        # 1. HISTORY FEEDBACK first: per-row reduction, then per-case.
        block = context.get("prediction_execution_pairs")
        if isinstance(block, dict) and block.get("pairs"):
            while current["tokens"] > max_tokens:
                block_size = _size(block)
                excess_chars = max(
                    200, int((current["tokens"] - max_tokens)
                             * ASCII_CHARS_PER_TOKEN))
                target = max(1200, block_size - excess_chars)
                if target >= block_size:
                    target = block_size - 400
                trimmed, info = trim_history_block(block, target,
                                                    stage="full_request")
                if _size(trimmed) >= block_size:
                    trimmed = _history_summary(block)
                    info["summarised"] = True
                context["prediction_execution_pairs"] = trimmed
                block = trimmed
                stages.append({**info, "stage": "history_feedback"})
                current = measured()
                if not trimmed.get("pairs"):
                    break
        # 2. Other non-core material (capability, retrieval, H+ notes) —
        # cheaper to lose than the calibration/evidence blocks below.
        noncore_keys = ("harness_capability", "capability_version",
                        "retrieval_evidence", "hplus_feedback",
                        "prediction_reminders", "notes")
        for key in noncore_keys:
            if current["tokens"] <= max_tokens:
                break
            value = context.get(key)
            if value in (None, {}, [], ""):
                continue
            before_chars = _size(value)
            kind = ("capability" if key == "harness_capability"
                    else "retrieval_evidence" if key == "retrieval_evidence"
                    else "generic")
            if isinstance(value, dict):
                trimmed, _info = trim_evidence_block(
                    value, max(200, int(before_chars * 0.5)), kind=kind)
            elif isinstance(value, list):
                keep = max(0, len(value) // 2)
                trimmed = value[:keep]
            else:
                trimmed = value
            context[key] = trimmed
            stages.append({"stage": f"noncore:{key}",
                           "before_chars": before_chars,
                           "after_chars": _size(trimmed)})
            current = measured()
        # 3. G/C/R evidence blocks (calibration facts: keep them longer).
        for key in ("implementation_r_evidence", "cost_evidence"):
            if current["tokens"] <= max_tokens:
                break
            value = context.get(key)
            if not isinstance(value, dict) or not value:
                continue
            before_chars = _size(value)
            target = max(300, int(before_chars * 0.5))
            trimmed, info = trim_evidence_block(value, target, kind=key)
            context[key] = trimmed
            stages.append({"stage": f"gcr_evidence:{key}",
                           "before_chars": before_chars,
                           "after_chars": _size(trimmed),
                           **info})
            current = measured()
        # Repeated halving of whatever remains outside the protected core.
        rounds = 0
        while current["tokens"] > max_tokens and rounds < 10:
            rounds += 1
            progressed = False
            for key in ("harness_capability", "capability_version",
                        "retrieval_evidence", "hplus_feedback",
                        "prediction_reminders", "notes",
                        "g_candidate_evidence",
                        "implementation_r_evidence", "cost_evidence"):
                if current["tokens"] <= max_tokens:
                    break
                value = context.get(key)
                if value in (None, {}, [], ""):
                    continue
                before_chars = _size(value)
                if isinstance(value, dict):
                    kind = ("capability" if key == "harness_capability"
                            else "retrieval_evidence" if key
                            == "retrieval_evidence" else key
                            if key in ("g_candidate_evidence", "cost_evidence",
                                       "implementation_r_evidence")
                            else "generic")
                    target = max(120, int(before_chars * 0.6))
                    trimmed, _info = trim_evidence_block(value, target,
                                                         kind=kind)
                elif isinstance(value, list):
                    trimmed = value[:max(0, len(value) - 1)]
                else:
                    trimmed = value[:max(0, len(value) - 100)]
                if _size(trimmed) >= before_chars:
                    continue
                context[key] = trimmed
                progressed = True
                stages.append({"stage": f"noncore_round:{key}",
                               "before_chars": before_chars,
                               "after_chars": _size(trimmed)})
                current = measured()
            if not progressed:
                break
        # 3b. G/C/R evidence at its irreducible minimum: keep the grades and
        # counts, replace per-case bodies with a reference. This is the last
        # step before the protected core is declared unsendable.
        for key in ("g_candidate_evidence", "implementation_r_evidence",
                    "cost_evidence"):
            if current["tokens"] <= max_tokens:
                break
            value = context.get(key)
            if not isinstance(value, dict) or not value:
                continue
            before_chars = _size(value)
            summary = _gcr_evidence_summary(key, value)
            if _size(summary) >= before_chars:
                continue
            context[key] = summary
            stages.append({"stage": "gcr_evidence:summary",
                           "block": key,
                           "before_chars": before_chars,
                           "after_chars": _size(summary)})
            current = measured()
        # 3c. Optional non-core context at its irreducible minimum. The
        # stored context and `orx inspect` keep the full material; the model
        # gets a named pointer rather than an oversized request.
        for key in ("prediction_reminders", "hplus_feedback", "notes",
                    "capability_version", "harness_capability",
                    "retrieval_evidence"):
            if current["tokens"] <= max_tokens:
                break
            value = context.get(key)
            if value in (None, {}, [], ""):
                continue
            before_chars = _size(value)
            context[key] = {
                "omitted_for_size": True,
                "detail_ref": "the stored prediction context",
                "note": ("this optional block was reduced to a reference to "
                         "keep the full request within its bound; the stored "
                         "context keeps the complete value"),
            }
            stages.append({"stage": "noncore:summary", "block": key,
                           "before_chars": before_chars,
                           "after_chars": _size(context[key])})
            current = measured()
        # 4. History at its irreducible minimum: counts-only report.
        block = context.get("prediction_execution_pairs")
        if current["tokens"] > max_tokens and isinstance(block, dict) \
                and (block.get("pairs") or block.get("same_task_comparisons")):
            context["prediction_execution_pairs"] = _history_summary(block)
            stages.append({"stage": "history_feedback:summary",
                           "summarised": True})
            current = measured()
    core = _core_sizes(request, system_prompt)
    over = current["tokens"] > max_tokens
    block: Dict[str, Any] = {
        "chars": current["chars"],
        "body_chars": current["body_chars"],
        "system_prompt_chars": current["system_prompt_chars"],
        "context_fragment_chars": _size(
            request.get("prediction_context") or {}),
        "estimator": current["estimator"],
        "token_estimator": current["estimator"],
        "chars_estimator": (
            "characters = system prompt + JSON body string length; the "
            "chars/4 conversion is a LEGACY bound only"),
        "tokens": current["tokens"],
        "token_basis": current["basis"],
        "bound_tokens": max_tokens,
        "bound_chars": budget["char_bound"],
        "budget": {
            "budget_tokens": budget["declared_tokens"],
            "reserved_output_tokens": budget["reserved_output_tokens"],
            "input_tokens": max_tokens,
            "chars_per_token": budget["chars_per_token"],
            "bound_chars": budget["char_bound"],
            "source": budget["source"],
            "estimator": current["estimator"],
        },
        "within_bound": not over,
        "sendable": not over,
        "before": {"tokens": before["tokens"], "chars": before["chars"]},
        "core": core,
        "n_trim_stages": len(stages),
        "trimming": stages,
        "wire": {
            "measured_object": "the request WITHOUT request_size",
            "excluded_keys": list(WIRE_EXCLUDED_REQUEST_KEYS),
            "note": ("request_size is LOCAL diagnostics: the provider "
                     "receives the request without it, and the budget "
                     "measures that exact wire object"),
        },
    }
    if over:
        block["over_budget"] = {
            "excess_tokens": current["tokens"] - max_tokens,
            "excess_chars": current["chars"] - int(
                budget["char_bound"]),
            "core_chars": core["core_chars"],
            "note": ("the WHOLE request exceeds the deployment's input "
                     "bound even after the history feedback and other "
                     "non-core blocks were reduced; the request is NOT sent. "
                     "The protected core is the system prompt + the current "
                     "task + the candidate + the execution constraints + "
                     "the prediction contract. Raise "
                     "OR_HARNESS_INPUT_BUDGET_TOKENS, shorten the task/"
                     "system prompt, or narrow the history window."),
        }
    return block
