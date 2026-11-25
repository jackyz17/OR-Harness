"""Small projections of existing actions/receipts; no independent event store.

Host reports are explicit observations, never inferred from commands, exception
words or transcripts. Formal receipts and host reports share event identities.
Coverage is a separate affirmative report: silence is not a clean implementation.
"""
from __future__ import annotations

import copy
import math


def timestamp(value):
    return (float(value) if isinstance(value, (int, float))
            and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None)


def formal_event_id(execution_id):
    return "execution:" + str(execution_id) + ":technical"


def episode_actions(harness, task_id, episode_id):
    # ActionLog's None episode filter is a wildcard. Implementation/cost
    # attribution requires an exact episode, including the episode-less one.
    return [a for a in harness.actions.query(task_id=task_id)
            if str(a.episode_id or "") == str(episode_id or "")]


def _refs(value, name):
    if not isinstance(value, list) or not value or any(
            not isinstance(v, str) or not v.strip() or len(v) > 1000 for v in value):
        raise ValueError(name + " requires nonempty evidence references")
    return list(dict.fromkeys(value))


def normalize_report(harness, outcome, *, task_id, episode_id, ended_at):
    """Validate the optional extensions to report_action's existing outcome.

    A selection is a real choose_next action for the scored winner. It need
    not have an execution. This does not claim/consume a formal execution.
    """
    outcome = copy.deepcopy(outcome)
    facts = outcome.get("implementation_facts", [])
    if not isinstance(facts, list) or len(facts) > 32:
        raise ValueError("implementation_facts must be a concise list (at most 32)")

    def selection(value):
        action = harness.actions.get(value)
        if (not action or action.action_type != "select_strategy"
                or str(action.task_id) != str(task_id)
                or str(action.episode_id or "") != str(episode_id or "")
                or (action.outcome or {}).get("kind") != "plan_next_choice"
                or not (action.outcome or {}).get("selected")
                or (action.outcome or {}).get("rejected")):
            raise ValueError("implementation evidence requires a real selected choice action in this episode")
        prediction = harness.strategy_predictions.get(action.outcome.get("prediction_id"))
        decision = harness.actions.get(action.parent_action_id) if action.parent_action_id else None
        if (not prediction or not decision or (decision.params or {}).get("kind") != "plan_next"
                or decision.action_id != prediction.trace.model_info.get("decision_action_id")
                or decision.outcome.get("selected_prediction_id") != prediction.prediction_id):
            raise ValueError("selection must belong to the actual scored planning decision")
        return action

    for fact in facts:
        if not isinstance(fact, dict):
            raise ValueError("implementation fact must be an object")
        choice = selection(fact.get("selection_action_id"))
        pid = choice.outcome["prediction_id"]
        if fact.get("prediction_id", pid) != pid:
            raise ValueError("implementation fact contradicts selected prediction")
        fact.update(prediction_id=pid, task_id=str(task_id), episode_id=episode_id)
        if not isinstance(fact.get("event_id"), str) or not fact["event_id"].strip():
            raise ValueError("implementation fact requires a stable event_id")
        at = timestamp(fact.get("occurred_at"))
        if at is None or at < choice.started_at or at > ended_at:
            raise ValueError("fact must occur after selection and before report end")
        if fact.get("phase") not in ("implementation", "pre_execute", "execution", "technical_verification"):
            raise ValueError("invalid implementation fact phase")
        if fact.get("classification") not in ("implementation", "environment", "framework", "acceptance", "mathematical", "other", "unknown"):
            raise ValueError("invalid implementation fact classification")
        for key in ("confirmed", "requires_correction", "obstructed", "excluded"):
            if key in fact and type(fact[key]) is not bool:
                raise ValueError(key + " must be boolean")
        fact["evidence_refs"] = _refs(fact.get("evidence_refs"), "fact")
        if not isinstance(fact.get("observer"), str) or not fact["observer"].strip():
            raise ValueError("host fact requires observer provenance")
        fact["source"] = "host_report"
        eid = fact.get("execution_id")
        if eid:
            record = harness.bank.get_pending(eid) or harness.bank.get(eid)
            action = harness.actions.get(getattr(record, "action_id", None)) if record else None
            if not action:
                action = next((a for a in harness.actions.query(task_id=task_id, episode_id=episode_id)
                               if a.linked_execution_id == eid), None)
            if (not record or not action or
                    (action.params or {}).get("decision_choice_action_id") != choice.action_id):
                raise ValueError("formal error reference does not belong to this selection")
            terminal = (record.execution_features.get("technical_outcome") or {}).get("label") == 1
            formal_error = fact.get("formal_error", terminal)
            if type(formal_error) is not bool:
                raise ValueError("formal_error must be boolean")
            fact["formal_error"] = formal_error
            if formal_error:
                fact["event_id"] = formal_event_id(eid)
            elif fact["event_id"].startswith("execution:"):
                raise ValueError("reserved formal identity cannot name a separate recovered defect")
            if not action.started_at <= at <= action.ended_at:
                at = action.ended_at
                fact["occurred_at"] = at
            fact["phase"] = "execution"
        elif fact["event_id"].startswith("execution:"):
            raise ValueError("reserved formal event identity requires execution_id")
        next_decisions = [a.started_at for a in episode_actions(harness, task_id, episode_id)
                          if a.action_type == "select_strategy" and (a.params or {}).get("kind") == "plan_next"
                          and a.started_at >= choice.started_at and a.action_id != choice.parent_action_id]
        if next_decisions and fact["occurred_at"] >= min(next_decisions):
            raise ValueError("implementation fact is outside the selected candidate window")
        repair = fact.get("correction")
        if repair is not None:
            if not isinstance(repair, dict) or not isinstance(repair.get("round_id"), str) or not repair["round_id"].strip():
                raise ValueError("started correction requires a stable round_id")
            start = timestamp(repair.get("started_at"))
            if start is None or not at <= start <= ended_at:
                raise ValueError("correction needs its actual start time after the defect")
            repair["evidence_refs"] = _refs(repair.get("evidence_refs"), "correction")
            if not isinstance(repair.get("method"), str) or not repair["method"].strip():
                raise ValueError("correction requires a concise method")
        # Bound the fact itself, not just the number of events. Full versions
        # and diffs live in artifacts/host logs referenced by evidence_refs.
        import json
        if len(json.dumps(fact, ensure_ascii=False)) > 6000:
            raise ValueError("implementation fact too large; use artifact references")
    for key in ("implementation_coverage", "correction_coverage"):
        value = outcome.get(key)
        if value is None:
            continue
        if not isinstance(value, dict) or type(value.get("complete")) is not bool:
            raise ValueError(key + " requires explicit complete boolean")
        start, stop = timestamp(value.get("from")), timestamp(value.get("through"))
        if start is None or stop is None or not start <= stop <= ended_at:
            raise ValueError(key + " requires valid from/through boundaries before report end")
        value["evidence_refs"] = _refs(value.get("evidence_refs"), key)
        if not isinstance(value.get("observer"), str) or not value["observer"].strip():
            raise ValueError(key + " requires observer provenance")
        value["source"] = "host_report"
        if key == "implementation_coverage":
            choice = selection(value.get("selection_action_id"))
            if abs(start - choice.started_at) > 1e-6:
                raise ValueError("implementation coverage must start at the actual selection")
            for flag in ("implemented", "candidate_complete"):
                if type(value.get(flag)) is not bool:
                    raise ValueError("implementation coverage requires " + flag + " boolean")
    return outcome


def choices(actions, prediction_id=None):
    return sorted((a for a in actions if a.action_type == "select_strategy"
                   and (a.outcome or {}).get("kind") == "plan_next_choice"
                   and (a.outcome or {}).get("selected")
                   and not (a.outcome or {}).get("rejected")
                   and (prediction_id is None or a.outcome.get("prediction_id") == prediction_id)),
                  key=lambda a: (a.started_at, a.action_id))


def collect_facts(actions, records=(), *, cutoff=None):
    """Deduplicate real event identities, preserving provenance and repairs."""
    found = {}
    for action in actions:
        if action.source == "hypothetical" or (cutoff is not None and
                (action.ended_at is None or action.ended_at > cutoff)):
            continue
        for raw in (action.outcome or {}).get("implementation_facts") or []:
            fact = copy.deepcopy(raw)
            fact["source_action_ids"] = [action.action_id]
            fact["provenance"] = [{"source": "host_report", "observer": fact.get("observer"),
                                   "action_id": action.action_id}]
            _merge(found, fact)
    from or_harness.world_model.r_signal import technical_attempt_fact
    for record in records:
        action = next((a for a in actions if a.linked_execution_id == record.execution_id
                       and a.action_type == "execute_strategy"), None)
        if not action or action.ended_at is None or (cutoff is not None and action.ended_at > cutoff):
            continue
        formal = technical_attempt_fact(record)
        if formal["label"] != 1:
            continue
        _merge(found, {"event_id": formal_event_id(record.execution_id),
            "execution_id": record.execution_id,
            "selection_action_id": (action.params or {}).get("decision_choice_action_id"),
            "prediction_id": (action.params or {}).get("prediction_id"),
            "occurred_at": action.ended_at, "phase": "execution",
            "classification": "implementation", "confirmed": True,
            "requires_correction": True, "summary": formal["reason"],
            "source": formal["source"], "evidence_refs": [record.execution_id],
            "provenance": [{"source": formal["source"], "execution_id": record.execution_id}],
            "source_action_ids": [action.action_id]})
    return sorted(found.values(), key=lambda f: (f["occurred_at"], f["event_id"]))


def _merge(found, fact):
    key = fact["event_id"]
    fact["corrections"] = [fact.pop("correction")] if fact.get("correction") else []
    fact["selection_action_ids"] = [fact.get("selection_action_id")]
    if key not in found:
        found[key] = fact
        return
    saved = found[key]
    # Source disagreement is not resolved by trusting the latest report.
    for field in ("selection_action_id", "classification"):
        if saved.get(field) != fact.get(field):
            saved.setdefault("conflicts", []).append(field)
    for field in ("evidence_refs", "source_action_ids", "provenance", "corrections", "selection_action_ids"):
        for item in fact.get(field) or []:
            if item not in saved.setdefault(field, []):
                saved[field].append(item)
    saved["confirmed"] = saved.get("confirmed") is True or fact.get("confirmed") is True
    saved["requires_correction"] = saved.get("requires_correction") is True or fact.get("requires_correction") is True


def implementation_window(prediction, actions, records=(), *, cutoff=None):
    selected = choices(actions, prediction.prediction_id)
    choice = selected[0] if len(selected) == 1 else None
    start = choice.started_at if choice else None
    subsequent = [a.started_at for a in actions if choice and
                  a.action_type == "select_strategy" and (a.params or {}).get("kind") == "plan_next"
                  and a.started_at >= start and a.action_id != choice.parent_action_id]
    coverages = [(a, (a.outcome or {}).get("implementation_coverage")) for a in actions
                 if a.source != "hypothetical" and a.ended_at is not None
                 and (cutoff is None or a.ended_at <= cutoff)]
    coverages = [(a, c) for a, c in coverages if c and choice and
                 c.get("selection_action_id") == choice.action_id]
    ends = subsequent + [c["through"] for _, c in coverages if c.get("candidate_complete")]
    end = min(ends) if ends else cutoff
    all_facts = collect_facts(actions, records, cutoff=cutoff)
    facts = [f for f in all_facts if choice and choice.action_id in f.get("selection_action_ids", [])
             and start <= f["occurred_at"] and (end is None or f["occurred_at"] <= end)]
    positives = [f for f in facts if f.get("confirmed") is True and not f.get("conflicts")
                 and not f.get("excluded") and f.get("classification") == "implementation"
                 and (f.get("requires_correction") or f.get("obstructed"))]
    complete = bool(ends) and any(c.get("complete") and c.get("implemented")
                     and c["from"] <= start and c["through"] >= end for _, c in coverages)
    ambiguous = any(not f.get("excluded") and (f.get("conflicts") or
                    f.get("confirmed") is not True or f.get("classification") == "unknown") for f in facts)
    label = 1 if positives else 0 if complete and not ambiguous else None
    from or_harness.world_model.contracts import R_OBSERVATION_VERSION
    return {"observation_rule_version": R_OBSERVATION_VERSION,
            "scope": "selected_candidate_implementation_window", "label": label,
            "reason": ("confirmed substantive implementation defect" if positives else
                       "complete observed implementation with no qualifying defect" if label == 0 else
                       "implementation coverage or selection attribution is incomplete"),
            "selection_action_id": choice.action_id if choice else None,
            "prediction_id": prediction.prediction_id, "start": start, "end": end,
            "complete": complete, "facts": facts,
            "coverage_refs": [a.action_id for a, _ in coverages],
            "evidence_refs": sorted({r for f in facts for r in f.get("evidence_refs") or []})}


def corrective_rounds(actions, records, *, anchor, end):
    # The actual round start, not the report delivery time, defines this
    # window. A late sourced report can complete an earlier closed span.
    facts = collect_facts(actions, records)
    rounds, gaps = {}, []
    starts = {}
    for fact in facts:
        if fact.get("confirmed") is True and not fact.get("excluded"):
            for correction in fact.get("corrections") or []:
                starts.setdefault(correction["round_id"], set()).add(correction["started_at"])
    for fact in facts:
        if fact.get("excluded"):
            continue
        for correction in fact.get("corrections") or []:
            start = timestamp(correction.get("started_at"))
            if start is not None and anchor is not None and start >= anchor and (end is None or start <= end):
                if fact.get("confirmed") is not True:
                    gaps.append("correction has an unconfirmed defect/trigger: " + fact["event_id"])
                    continue
                rid = correction["round_id"]
                if len(starts.get(rid) or []) > 1:
                    gaps.append("conflicting start for correction round " + rid)
                    continue
                if rid in rounds and rounds[rid]["started_at"] != start:
                    gaps.append("conflicting start for correction round " + rid)
                rounds[rid] = {"round_id": rid, "started_at": start,
                    "event_id": fact["event_id"], "source_refs": fact.get("source_action_ids") or []}
    # Formal retries remain an automatic lower bound. A repair followed by
    # resubmission is one continuous round, not two charges.
    by_id = {r.execution_id: r for r in records}
    for action in actions:
        if (action.action_type == "execute_strategy" and action.source != "hypothetical"
                and anchor is not None and action.started_at >= anchor
                and (end is None or action.started_at <= end)
                and (action.outcome or {}).get("phase") != "association"
                and (not action.linked_execution_id or action.linked_execution_id not in by_id)):
            gaps.append("formal action has no recoverable receipt: " + action.action_id)
    attempts = sorted((a for a in actions if a.action_type == "execute_strategy"
                       and a.linked_execution_id in by_id and a.source != "hypothetical"
                       and (end is None or a.started_at <= end)), key=lambda a: (a.started_at, a.action_id))
    unique, seen = [], set()
    for action in attempts:
        if action.linked_execution_id not in seen:
            unique.append(action)
            seen.add(action.linked_execution_id)
    attempts = unique
    for previous, current in zip(attempts, attempts[1:]):
        record = by_id[previous.linked_execution_id]
        acceptance_failures = [a for a in actions if a.action_type == "verify"
            and a.linked_execution_id == record.execution_id and a.source != "hypothetical"
            and a.ended_at is not None and a.ended_at <= current.started_at
            and (a.outcome or {}).get("state") == "failed"]
        failed = bool(record.failures) or (record.quality or {}).get("status") in ("error", "timeout") or bool(acceptance_failures)
        if not failed or anchor is None or current.started_at < anchor:
            continue
        matching = [r for r in rounds.values() if previous.ended_at is not None
                    and previous.ended_at <= r["started_at"] <= current.started_at]
        if matching:
            continue
        # A round started before this forecast is not charged again at its
        # later formal resubmission. Inspect the same event facts pre-anchor.
        prior_repairs = [c for f in facts if f.get("confirmed") is True and not f.get("excluded")
                         for c in f.get("corrections") or []
                         if previous.ended_at is not None and previous.ended_at <= c["started_at"] <= current.started_at]
        if prior_repairs:
            continue
        rid = "formal-retry:" + current.action_id
        rounds[rid] = {"round_id": rid, "started_at": current.started_at,
                       "source_refs": [previous.action_id, current.action_id]}
    coverage = [a for a in actions if a.source != "hypothetical" and a.ended_at is not None
                and end is not None
                and (a.outcome or {}).get("correction_coverage", {}).get("complete")
                and anchor is not None
                and a.outcome["correction_coverage"]["from"] <= anchor
                and a.outcome["correction_coverage"]["through"] >= end - 1e-6]
    return {"rounds": sorted(rounds.values(), key=lambda r: (r["started_at"], r["round_id"])),
            "complete": bool(coverage) and not gaps, "gaps": gaps,
            "coverage_refs": [a.action_id for a in coverage]}


def execution_implementation_evidence(harness, record, *, cutoff=None):
    """Read-time projection: never rewrite an archived execution record."""
    pid = (record.execution_features or {}).get("prediction_id")
    prediction = harness.strategy_predictions.get(pid) if pid else None
    if not prediction:
        return {}
    from or_harness.world_model.contracts import R_OBSERVATION_VERSION
    if (prediction.trace.model_info.get("risk_convention") or {}).get("observation_rule_version") != R_OBSERVATION_VERSION:
        return {"coverage": "legacy formal receipt only; no implementation-window negative inferred"}
    actions = episode_actions(harness, record.task_id, prediction.candidate.episode_id)
    value = implementation_window(prediction, actions, [record], cutoff=cutoff)
    # Preserve pre-execute abandoned candidates in the later real task chain,
    # as contextual evidence only. Their errors never label this candidate.
    earlier = [a for a in choices(actions) if value.get("start") is not None
               and a.started_at < value["start"] and not a.outcome.get("adopted_by_action_id")]
    prior = []
    for choice in earlier[-2:]:
        p = harness.strategy_predictions.get(choice.outcome["prediction_id"])
        if p:
            prior.append(compact_evidence(implementation_window(p, actions, cutoff=cutoff), max_facts=1))
    if prior:
        value["prior_selected_implementations"] = {"role": "same-episode context only; excluded from this candidate's R label",
                                                   "windows": prior, "omitted_windows": max(0, len(earlier)-2)}
    return value


def compact_evidence(value, *, max_facts=3):
    """Bounded send-time view, with exact actions for full versions/diffs."""
    out = {k: copy.deepcopy(value[k]) for k in (
        "observation_rule_version", "scope", "label", "reason", "selection_action_id",
        "prediction_id", "start", "end", "complete", "coverage", "coverage_refs",
        "prior_selected_implementations") if k in value}
    facts = value.get("facts") or []
    out["n_facts"] = len(facts)
    out["facts"] = []
    for fact in facts[:max_facts]:
        item = {k: copy.deepcopy(fact[k]) for k in (
            "event_id", "occurred_at", "phase", "classification", "confirmed",
            "requires_correction", "obstructed", "excluded", "source", "observer",
            "conflicts", "formal_error") if k in fact}
        item["summary"] = str(fact.get("summary") or "")[:240]
        item["evidence_refs"] = (fact.get("evidence_refs") or [])[:4]
        item["detail_refs"] = ["orx inspect --bank actions --action " + aid
                               for aid in (fact.get("source_action_ids") or [])[:3]]
        item["corrections"] = [{**{k: copy.deepcopy(c[k]) for k in (
            "round_id", "started_at", "result", "evidence_refs", "verification_refs") if k in c},
            "method": str(c.get("method") or "")[:240]} for c in (fact.get("corrections") or [])[:3]]
        out["facts"].append(item)
    out["omitted_facts"] = max(0, len(facts) - max_facts)
    return out
