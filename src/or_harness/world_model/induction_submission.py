"""One compact agent submission for a required induction review.

The compact protocol is deliberately SYNTACTIC: the agent states the learning
result, the rule (when / do / expect), one evidence-and-argument object and a
publication intention.  The framework translates that object into the SAME
``relations`` / ``review`` payload the versioned writer and the H+ receipt
already consume -- it does not add a second knowledge lifecycle.

A compact submission never invents an agent judgement.  ``sufficient`` and
``substantive`` are read off the agent's declared result
(``new_rule``/``revision`` are substantive; ``already_covered`` /
``no_new_knowledge`` / ``evidence_added`` are not; ``insufficient_evidence``
is neither), while ``publication`` remains a separate choice and verification
is still whatever the existing writer actually checks.

The object shape is::

    {
      "result": "new_rule" | "revision" | "evidence_added" |
                "already_covered" | "no_new_knowledge" | "insufficient_evidence",
      "rule": {"when": "...", "do": "...", "expect": "..."},
      "entry_id": "...", "version": 2,          # revision / evidence_added
      "basis": {"kind": "derivation" | "recorded_checks" |
                        "artifact_inspection",
                "refs": ["EX", "rule_fact:...", "artifacts.submitted_script"],
                "analysis": "short substantive argument"},
      "publication": {"state": "published" | "candidate", "reason": "..."}
    }

``rule``/``entry_id``/``version``/``publication`` are only legal for the
results that use them.  ``basis.refs`` is the single place actual evidence is
named; the framework keeps every reference for the learning comparison and
derives the execution identities needed by the versioned writer.  A
value reference such as ``rule_fact:...`` is resolved to its recorded
execution id; an ``artifacts.*`` / ``task_check.*`` reference is bound to the
trigger execution of the induction job.  Nothing here writes or verifies a
fact -- those steps stay in ``rule_versions`` and ``hplus``.
"""
from __future__ import annotations

import copy

from or_harness.core.storage import StorageError
from or_harness.world_model.conditions import get_value

PROTOCOL_VERSION = "wm-induction-submission/1"

NEW_RULE = "new_rule"
REVISION = "revision"
EVIDENCE_ADDED = "evidence_added"
ALREADY_COVERED = "already_covered"
NO_NEW_KNOWLEDGE = "no_new_knowledge"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"

RESULTS = (NEW_RULE, REVISION, EVIDENCE_ADDED, ALREADY_COVERED,
           NO_NEW_KNOWLEDGE, INSUFFICIENT_EVIDENCE)
# The legacy review outcome names are accepted as aliases; the canonical
# names above are what the receipt and documentation use.
_RESULT_ALIASES = {
    "revised": REVISION,
    "no_increment": NO_NEW_KNOWLEDGE,
    "insufficient_material": INSUFFICIENT_EVIDENCE,
}

#: Results that write a relation (and therefore need at least one recorded
#: execution identity).
WRITE_RESULTS = (NEW_RULE, REVISION, EVIDENCE_ADDED)
#: Results whose compact payload is a supported conclusion over the cited
#: basis.  ``insufficient_evidence`` deliberately is NOT here.
SUPPORTED_RESULTS = (NEW_RULE, REVISION, EVIDENCE_ADDED, ALREADY_COVERED,
                     NO_NEW_KNOWLEDGE)
#: Results that claim a substantive increment.  This is the agent's learning
#: result, translated; the framework does not upgrade a non-increment result.
SUBSTANTIVE_RESULTS = (NEW_RULE, REVISION)

BASIS_KINDS = ("derivation", "recorded_checks", "artifact_inspection")

#: The old review outcome each compact result means.  One table, used both
#: for normalisation and for documentation.
OUTCOME = {
    NEW_RULE: "new_rule",
    REVISION: "revised",
    EVIDENCE_ADDED: "evidence_added",
    ALREADY_COVERED: "already_covered",
    NO_NEW_KNOWLEDGE: "no_increment",
    INSUFFICIENT_EVIDENCE: "insufficient_material",
}

_TOP_LEVEL = {"result", "rule", "entry_id", "version", "basis", "publication",
              "hypothesis_outcome", "deviation_reason"}
_RULE_KEYS = {"when", "do", "expect", "effect_scope", "subject"}
_BASIS_KEYS = {"kind", "refs", "analysis"}


def _text(value, field, *, required=True):
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value.strip()


def _unknown_keys(raw, allowed, field):
    extra = sorted(set(raw) - set(allowed))
    if extra:
        raise ValueError(f"{field} has unknown field(s) {extra}; allowed: "
                         + ", ".join(sorted(allowed)))


def _result(raw):
    value = raw.get("result")
    if isinstance(value, str):
        value = _RESULT_ALIASES.get(value, value)
    if not isinstance(value, str) or value not in RESULTS:
        raise ValueError("submission.result must be one of " + ", ".join(RESULTS))
    return value


def _rule(raw, result):
    value = raw.get("rule")
    if result not in (NEW_RULE, REVISION):
        if value is not None:
            raise ValueError("submission.rule is only valid for new_rule or revision")
        return None
    if value is None:
        raise ValueError("submission.rule is required for " + result)
    if not isinstance(value, dict):
        raise ValueError("submission.rule must be a JSON object")
    _unknown_keys(value, _RULE_KEYS, "submission.rule")
    body = {"when": _text(value.get("when"), "submission.rule.when"),
            "do": _text(value.get("do"), "submission.rule.do"),
            "expect": _text(value.get("expect"), "submission.rule.expect")}
    if value.get("effect_scope") is not None:
        body["effect_scope"] = _text(value.get("effect_scope"),
                                     "submission.rule.effect_scope")
    if value.get("subject") is not None:
        body["subject"] = _text(value.get("subject"),
                                "submission.rule.subject")
    return body


def _entry(raw, result):
    entry_id = raw.get("entry_id")
    version = raw.get("version")
    needs_entry = result in (REVISION, EVIDENCE_ADDED)
    if not needs_entry:
        if entry_id is not None:
            raise ValueError("submission.entry_id is only valid for revision or evidence_added")
        if version is not None:
            raise ValueError("submission.version is only valid with submission.entry_id")
        return None, None
    if entry_id is None or isinstance(entry_id, bool) or not isinstance(entry_id, (str, int)) \
            or not str(entry_id).strip():
        raise ValueError("submission.entry_id is required for " + result)
    if version is None:
        return str(entry_id).strip(), None
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("submission.version must be a positive integer")
    return str(entry_id).strip(), version


def _basis(raw, result):
    value = raw.get("basis")
    if value is None:
        if result == INSUFFICIENT_EVIDENCE:
            return None
        raise ValueError("submission.basis is required for " + result)
    if not isinstance(value, dict):
        raise ValueError("submission.basis must be a JSON object")
    _unknown_keys(value, _BASIS_KEYS, "submission.basis")
    kind = value.get("kind")
    if kind not in BASIS_KINDS:
        raise ValueError("submission.basis.kind must be one of " + ", ".join(BASIS_KINDS))
    refs = value.get("refs")
    if not isinstance(refs, list) or not refs:
        raise ValueError("submission.basis.refs must be a nonempty list of references")
    clean = []
    for index, ref in enumerate(refs):
        clean.append(_text(ref, f"submission.basis.refs[{index}]"))
    # Preserve the agent's order; duplicates are one evidence identity.
    refs = list(dict.fromkeys(clean))
    analysis = _text(value.get("analysis"), "submission.basis.analysis")
    return {"kind": kind, "refs": refs, "analysis": analysis}


def _publication(raw, result):
    value = raw.get("publication")
    if value is None:
        return None
    if result not in WRITE_RESULTS:
        raise ValueError("submission.publication is only valid for a written rule")
    if not isinstance(value, dict):
        raise ValueError("submission.publication must be a JSON object with state and reason")
    _unknown_keys(value, {"state", "reason", "scope"}, "submission.publication")
    state = value.get("state")
    if state not in ("published", "publish", "candidate"):
        raise ValueError("submission.publication.state must be published or candidate")
    return copy.deepcopy(value)


def _optional_hypothesis(raw):
    value = raw.get("hypothesis_outcome")
    if value is None:
        return "unknown"
    if value not in ("supported", "refuted", "unknown"):
        raise ValueError("submission.hypothesis_outcome must be supported, refuted or unknown")
    return value


def _job(harness, induction_id):
    if induction_id is None:
        return None
    from or_harness.world_model.hplus import get_job
    job = get_job(harness, induction_id)
    if job is None:
        raise ValueError("unknown induction job " + str(induction_id))
    return job


def _ref_execution_id(harness, ref, job):
    """Map one actual reference to a recorded execution identity, if it has one.

    This is an ASSOCIATION, not a verification: the existing writer still
    resolves every id against the bank, and the comparison still checks task
    identity, time scope and support.  Return None for references that name
    only frozen context (a snapshot) -- those cannot by themselves support a
    written rule.
    """
    if ref.startswith("execution:"):
        return ref.split(":", 1)[1]
    if ref.startswith(("artifacts.", "task_check.")):
        if not job or not job.get("execution_ids"):
            raise ValueError("evidence reference " + ref + " belongs to the "
                             "induction job's trigger execution; pass --induction-id")
        return str(job["execution_ids"][0])
    if ":" not in ref:
        return ref
    try:
        value = get_value(harness.store, ref)
    except StorageError:
        return None
    if not isinstance(value, dict):
        return None
    eid = value.get("execution_id")
    if eid is None and isinstance(value.get("record"), dict):
        eid = value["record"].get("execution_id")
    return str(eid) if eid is not None else None


def _execution_ids(harness, refs, job):
    ids = []
    for ref in refs:
        eid = _ref_execution_id(harness, ref, job)
        if eid:
            ids.append(eid)
    return list(dict.fromkeys(ids))


def _comparison(kind, refs, analysis, result, execution_ids):
    if execution_ids:
        source = ("compact induction submission (" + kind +
                  " basis over recorded executions " +
                  ", ".join(execution_ids) + ")")
    else:
        source = ("compact induction submission (" + kind +
                  " basis over cited references " + ", ".join(refs) + ")")
    return {
        "sufficient": result in SUPPORTED_RESULTS,
        "substantive": result in SUBSTANTIVE_RESULTS,
        "basis": kind,
        # A stable, input-derived provenance label.  Execution identities,
        # task ids, timestamps and rule versions are resolved and frozen by
        # the existing H+ comparison/receipt code; this field only names the
        # actual execution identities the compact submission associated with
        # the cited references.
        "source": source,
        "assessment": analysis,
        "evidence_refs": list(refs),
    }


def normalize_submission(harness, submission, *, induction_id=None):
    """Translate one compact submission into the existing writer payload.

    Returns ``{"relations": [...], "review": {...}, "result": <compact>,
    "protocol": PROTOCOL_VERSION}``.  It is read-only: no rule, receipt,
    event or status is written here.
    """
    if not isinstance(submission, dict):
        raise ValueError("submission must be a JSON object")
    _unknown_keys(submission, _TOP_LEVEL, "submission")
    result = _result(submission)
    rule = _rule(submission, result)
    entry_id, version = _entry(submission, result)
    basis = _basis(submission, result)
    publication = _publication(submission, result)
    hypothesis = _optional_hypothesis(submission)
    deviation = submission.get("deviation_reason")
    if deviation is not None:
        deviation = _text(deviation, "submission.deviation_reason")
    job = _job(harness, induction_id)
    refs = basis["refs"] if basis else []
    execution_ids = _execution_ids(harness, refs, job)
    if result in WRITE_RESULTS and not execution_ids:
        raise ValueError("submission.basis.refs must name at least one recorded "
                         "execution (or a fact/artifact reference that resolves to one) "
                         "for " + result)

    relations = []
    if result in WRITE_RESULTS:
        relation = {}
        if rule:
            relation.update(rule)
        if entry_id is not None:
            relation["entry_id"] = entry_id
        if version is not None:
            relation["version"] = version
        if result == REVISION:
            relation["operation"] = "revise"
        elif result == EVIDENCE_ADDED:
            relation["operation"] = "add_evidence"
        relation["evidence"] = list(execution_ids)
        if publication is not None:
            relation["publication"] = publication
        relations.append(relation)

    review = {
        "outcome": OUTCOME[result],
        "hypothesis_outcome": hypothesis,
    }
    if deviation is not None:
        review["deviation_reason"] = deviation
    if basis is not None:
        review["conclusion"] = basis["analysis"]
        review["comparison"] = _comparison(
            basis["kind"], basis["refs"], basis["analysis"], result,
            execution_ids)
    elif result == INSUFFICIENT_EVIDENCE:
        # No comparison is fabricated when the agent declares the evidence
        # insufficient.  The existing empty-review path records the unknown
        # learning label.
        review["conclusion"] = None
    return {"relations": relations, "review": review,
            "result": result, "protocol": PROTOCOL_VERSION}
