"""World-model providers: the library-side interface + one minimal adapter.

Layer split (the deployment owns the model, the library owns the contract):

- OR-Harness: input assembly, prediction contract, explicit invocation,
  result validation, frozen records, cost & feedback linkage.
- Provider adapter: hands the prediction request to a configured model
  capability, returns the model's raw result plus whatever usage/error
  information it exposes.
- Caller / deployment: model name, service address, credentials, timeout,
  output limits, call budget.

The default provider is :class:`NotConfiguredProvider` — with it, every
prediction request returns ``not_configured`` and NO network activity ever
happens. The existing recall / predict_cost / execute / record paths never
invoke a provider at all; a model call happens only through the explicit
``predict_outcome`` entry point.

:class:`HttpChatProvider` is the one minimal adapter (OpenAI-compatible
chat/completions over stdlib urllib). No multi-vendor routing, no model
registry, no automatic failover. Credentials are passed at construction
and never persisted: ``describe()`` reports only the model name and
non-sensitive configuration.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

#: Version of the request schema this module builds (mirrored into the
#: prediction's model_info so stored records stay interpretable).
PROTOCOL_VERSION = "openai-chat/1"

#: The system prompt sent to the model. The model is asked for STRUCTURED
#: JSON only; the framework validates everything. Field omissions are
#: preferred over placeholder values — an absent field stays unknown.
SYSTEM_PROMPT = (
    "You are a world model for an operations-research harness. Given the "
    "current information state (belief snapshot view) and ONE candidate "
    "action, predict the consequences of executing that action.\n"
    "OUTPUT RULES (read first, follow strictly):\n"
    "1. Respond with EXACTLY ONE raw JSON object and NOTHING else. No "
    "markdown, no code fences, no explanation before or after the JSON.\n"
    "2. Every field below is OPTIONAL. If you lack evidence for a field, "
    "OMIT it entirely — never write \"unknown\", null, -1, or a guess as a "
    "placeholder value.\n"
    "3. All numbers must be valid JSON numbers (not strings).\n"
    "4. You predict WHAT HAPPENS IF the action is executed. You have NOT "
    "executed it. Do NOT solve the problem: do not output an objective "
    "value, an optimal solution, or any concrete answer number. The only "
    "legitimate numeric predictions are quality, failure_prob, the cost "
    "dimensions, confidence, and uncertainty.\n"
    "Fields (all optional):\n"
    "- outcome_status: one of optimal|feasible|infeasible|unbounded|"
    "timeout|error (omit if you cannot judge)\n"
    "- feasible: boolean\n"
    "- quality: number in [0,1] — the expected quality of THIS "
    "execution's result, where 1.0 = the result matches the task's own "
    "optimum, 0.7-0.9 = good feasible, 0.4-0.6 = marginal/uncertain, "
    "below 0.4 = poor. Do NOT default to 0.5; omit the field when you "
    "have no quality-relevant evidence\n"
    "- failure_prob: number in [0,1]\n"
    "- cost: object containing only the dimensions you have evidence for, "
    "each a non-negative JSON number: {llm_tokens, tool_calls, "
    "solver_runtime_s, retries, latency_s}\n"
    "- expected_error_kinds: list of error categories you expect\n"
    "- state_changes: object describing the SHAPE of the successor task "
    "state you expect — which fields will change and in what category "
    "(e.g. \"an incumbent solution will exist\", \"the model artifact "
    "will be rewritten\", \"verification evidence will be added\"). "
    "Describe categories and shapes, NOT solved values: never put an "
    "objective value, an optimal solution, or a concrete answer inside "
    "this object\n"
    "- knowledge_changes: list of changes this action is expected to make "
    "to the harness's ACCUMULATED KNOWLEDGE (not to the task's solution). "
    "Omit the field entirely when you see no such change. Each item is an "
    "object:\n"
    "    {\n"
    "      \"target\": the knowledge it bears on. Either\n"
    "          {\"kind\": \"existing_entry\", \"entry_id\": \"<id from "
    "candidate_knowledge_targets>\", \"strategy_id\": \"<sid>\"} to say "
    "something about a claim that already exists (the entry_id MUST come "
    "from the targets you were given), or\n"
    "          {\"kind\": \"hypothesis\", \"strategy_id\": \"<sid>\"} to "
    "propose a claim that does not exist yet. A hypothesis is allowed even "
    "with no targets offered, but it THEN MUST also carry the two fields "
    "below,\n"
    "      \"change\": one of adds_evidence | supports | revises | refutes "
    "| candidate_forms | narrows,\n"
    "      \"horizon\": \"after_execution\" when the change is visible as "
    "soon as the action's evidence is recorded, or "
    "\"after_consolidation\" when it can only happen later, during offline "
    "consolidation,\n"
    "      \"expected_observation\": REQUIRED for a hypothesis - an object "
    "of the concrete fields you would expect to see in the recorded result "
    "(e.g. {\"feasible\": true}). Use only keys that appear on an execution "
    "record: status, feasible, objective, gap, optimality, or a named cost "
    "dimension such as llm_tokens. This is what the claim will be judged "
    "against, so a vague expectation cannot be confirmed,\n"
    "      \"check_condition\": REQUIRED for a hypothesis - one sentence "
    "stating how that observation would settle the claim,\n"
    "      \"preconditions\": optional list of what must happen for the "
    "change to be observable at all: independent_replication (evidence "
    "from a different task), contrast_execution (a companion run to "
    "compare against), verification_check (an admission verdict), or "
    "additional_measurement:<dimension>,\n"
    "      \"expected_extra_cost\": when the change requires extra spend "
    "beyond this action (a contrast run, a consolidation), the cost "
    "dimensions that spend will consume,\n"
    "      \"prediction_basis\": list of the evidence keys you relied on,\n"
    "      \"uncertainty\": number in [0,1]\n"
    "    }\n"
    "- confidence: number in [0,1] — your self-reported confidence "
    "(uncalibrated)\n"
    "- evidence_basis: list of the evidence keys you relied on\n"
    "- unsupported_fields: object of {field: reason} for fields you "
    "cannot or will not predict\n"
    "Do NOT fabricate evidence. If the provided state contains no relevant "
    "experience or knowledge for the action, say so in unsupported_fields "
    "and give low confidence. Output the JSON object only."
)

#: The system prompt for offline knowledge induction (M4).
SYSTEM_PROMPT_INDUCE = (
    "You are a world model for an operations-research knowledge bank. "
    "Given the current strategic knowledge state and a proposed evidence "
    "bundle for an induction or revision action, predict the consequences "
    "and long-term value of performing this induction.\n"
    "OUTPUT RULES (read first, follow strictly):\n"
    "1. Respond with EXACTLY ONE raw JSON object and NOTHING else. No "
    "markdown, no code fences, no explanation before or after the JSON.\n"
    "2. Every field below is OPTIONAL. Omit any field you lack evidence "
    "for — never write \"unknown\", null, or placeholders.\n"
    "3. All numbers must be valid JSON numbers.\n"
    "Fields (all optional):\n"
    "- candidate_formation_prob: number in [0,1] (probability a valid claim "
    "forms)\n"
    "- expected_reuse_benefit: number in [0,1] (expected incremental benefit "
    "per future matching task relative to current knowledge)\n"
    "- generalization_risk: number in [0,1] (risk the induced claim fails on "
    "future matching problems)\n"
    "- quality: number in [0,1] (expected quality level of the induced claim)\n"
    "- failure_prob: number in [0,1] (expected failure rate under the claim)\n"
    "- cost: object of {llm_tokens, tool_calls, solver_runtime_s, retries, "
    "latency_s} — expected per-task execution cost under this strategy\n"
    "- confidence: number in [0,1]\n"
    "- evidence_basis: list of evidence keys you relied on\n"
    "- unsupported_fields: object of {field: reason}\n"
    "Do NOT fabricate evidence. Output the JSON object only."
)


class ProviderError(Exception):
    """A provider call failed (timeout, HTTP error, bad payload).

    Carries whatever partial call cost is known — a failed call may still
    have consumed resources."""


_FENCE_RE = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n(?P<body>.*?)\n?\s*```\s*$", re.DOTALL)


def _strip_code_fence(content: str) -> str:
    """Extract the JSON body from a markdown code fence when the model
    wrapped its answer in ```json ... ``` despite instructions. Bare JSON
    passes through unchanged; prose around the fence is NOT salvaged (the
    error stays honest)."""
    match = _FENCE_RE.match(content)
    if match:
        return match.group("body").strip()
    return content.strip()


class WorldModelProvider:
    """The provider contract. ``predict`` receives the assembled request
    (input view + action spec + output guidance) and returns a dict:

    ``{"payload": <model JSON or None>, "usage": {...} | None,
       "error": str | None, "latency_s": float}``

    Subclasses implement the actual model call. ``describe`` returns the
    model identity and non-sensitive configuration summary.

    ``timeout_s`` (optional): the caller's remaining time budget for THIS
    call. A provider that can bound its own wait (e.g. an HTTP client)
    uses ``min(timeout_s, its own configured timeout)``; a synchronous
    provider that CANNOT be safely interrupted mid-call must declare that
    limitation honestly — the budget is a request, not a guarantee, and
    the caller re-checks the clock after the call returns."""

    name = "abstract"

    def predict(self, request: Dict[str, Any],
                timeout_s: Optional[float] = None) -> Dict[str, Any]:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"provider": self.name}


class NotConfiguredProvider(WorldModelProvider):
    """The default: predictions are not enabled. Every call returns the
    explicit ``not_configured`` result — no network, no model, no cost."""

    name = "not-configured"

    def predict(self, request: Dict[str, Any],
                timeout_s: Optional[float] = None) -> Dict[str, Any]:
        return {
            "payload": None,
            "usage": None,
            "error": "world-model provider is not configured: pass a "
                     "WorldModelProvider to ORHarness(world_model=...) to "
                     "enable outcome predictions",
            "latency_s": 0.0,
            "not_configured": True,
        }


class HttpChatProvider(WorldModelProvider):
    """Minimal OpenAI-compatible chat/completions adapter (stdlib only).

    Configuration comes entirely from the constructor — the deployment
    environment owns the model name, endpoint, credentials, and limits.
    The API key lives only in the request header; it is never persisted,
    logged, or included in ``describe()``."""

    name = "http-chat"

    def __init__(self, base_url: str, model: str, api_key: str, *,
                 timeout_s: float = 30.0, max_output_tokens: int = 2048,
                 temperature: float = 0.2):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.max_output_tokens = int(max_output_tokens)
        self.temperature = float(temperature)

    def describe(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "protocol": PROTOCOL_VERSION,
            "model": self.model,
            "base_url": self.base_url,
            "timeout_s": self.timeout_s,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            # No credentials here, ever.
        }

    def predict(self, request: Dict[str, Any],
                timeout_s: Optional[float] = None) -> Dict[str, Any]:
        """One explicit POST to {base_url}/chat/completions.

        Returns the parsed payload and usage, or a structured error with
        whatever partial cost is known. No retries are attempted here —
        the caller decides whether to re-issue (each real attempt's cost
        is recorded separately).

        ``timeout_s``: the caller's remaining time budget for this call.
        The effective socket timeout is ``min(timeout_s, self.timeout_s)``
        — the provider never waits longer than the caller's budget
        allows."""
        effective_timeout = self.timeout_s
        if timeout_s is not None:
            effective_timeout = max(0.001, min(float(timeout_s),
                                               self.timeout_s))
        action_type = (request.get("action_spec") or {}).get("action_type")
        prompt = (SYSTEM_PROMPT_INDUCE if action_type == "induce"
                  else SYSTEM_PROMPT)
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(
                    request, ensure_ascii=False, default=str)},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST")
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
                raw = resp.read().decode("utf-8", "replace")
            latency = time.monotonic() - started
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except Exception:
                pass
            raise ProviderError(
                f"HTTP {exc.code}: {detail or exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError(f"request failed: {exc}") from exc
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"non-JSON response: {exc}") from exc
        choices = envelope.get("choices") or []
        if not choices:
            raise ProviderError("response has no choices")
        content = choices[0].get("message", {}).get("content")
        usage = envelope.get("usage") or {}
        payload = None
        parse_error = None
        if isinstance(content, str) and content.strip():
            cleaned = _strip_code_fence(content)
            try:
                payload = json.loads(cleaned)
            except json.JSONDecodeError as exc:
                parse_error = f"model content is not valid JSON: {exc}"
        elif content is None:
            parse_error = "model returned no content"
        return {
            "payload": payload,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            } if usage else None,
            "error": parse_error,
            "latency_s": latency,
        }
