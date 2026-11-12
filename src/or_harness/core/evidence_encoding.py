"""Lossless codes for framework-fixed prose in new execution rows.

Custom explanations are never replaced. Full audit reads render these codes;
normal reading views use the structured fields and keep masks/scopes intact.
"""
from __future__ import annotations

import copy

NOTES = {
    "config_sources": "the configuration that actually took effect, split by observer: 'executor' keys are the ones the sandbox set (a script may not overwrite them), 'script_reported' keys are the values the solve script read back from the solver. A key absent here is UNKNOWN — never a copy of the predicted config",
    "configured_instruction": "these values were HANDED to the script by the executor (through the environment). The framework cannot observe whether the solver honoured them, so they are not evidence that the solver used them: only a value the script READ BACK appears under 'script_reported'",
    "config_conflict": "the script reported a value for a key the executor controls; the executor's own value was kept",
    "config_missing": "the solve script reported no config (or its receipt carried no matching action_id stamp): script-internal parameters stay unknown",
    "performed_receipt": "the method the solve script reports it ACTUALLY performed, stamped with this attempt's action id. It is an observation, kept separate from the planned method — a plan is not a performed method",
}
PATHS = {"config_sources": ("execution_config", "note"),
         "configured_instruction": ("execution_config", "executor_configured_note"),
         "config_conflict": ("execution_config", "conflicts_note"),
         "config_missing": ("execution_config", "receipt"),
         "performed_receipt": ("method_receipt", "note")}


def compact_payload(raw):
    value = copy.deepcopy(raw)
    features = value.get("execution_features") or {}
    codes = []
    for code, (block, field) in PATHS.items():
        row = features.get(block)
        if isinstance(row, dict) and row.get(field) == NOTES[code]:
            row.pop(field); codes.append(code)
    receipt = features.get("method_receipt")
    if isinstance(receipt, dict) and receipt.get("method") == value.get("method_actual") and receipt.get("method") is not None:
        receipt.pop("method"); receipt["method_ref"] = "method_actual"
    if codes:
        value["explanation_codes"] = codes
    value["evidence_encoding"] = "execution-evidence/2"
    return value


def expand_payload(raw):
    value = copy.deepcopy(raw)
    features = value.get("execution_features") or {}
    for code in value.get("explanation_codes") or []:
        if code in PATHS:
            block, field = PATHS[code]
            features.setdefault(block, {}).setdefault(field, NOTES[code])
    receipt = features.get("method_receipt")
    if isinstance(receipt, dict) and receipt.get("method_ref") == "method_actual":
        receipt.setdefault("method", copy.deepcopy(value.get("method_actual")))
        receipt.pop("method_ref", None)
    return value
