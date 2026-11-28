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
happens. Prediction calls use only the explicit ``wm-so/1`` strategy-outcome
service. Recall, cost diagnostics, execution and recording make no model call.

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


def _truthy_env(name: str) -> bool:
    """True only for an explicit opt-in value of ``name``.

    Accepted spellings are ``1``, ``true``, ``yes`` and ``on``
    (case-insensitive); unset, empty, ``0``, ``false``, ``no``, ``off`` and
    anything else are False. There is no "maybe": an unrecognised value
    leaves the default in force rather than half-enabling a behaviour, so a
    typo cannot silently change the wire format."""
    import os
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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

    def count_tokens(self, text: str) -> Optional[int]:
        """The provider's OWN token count for ``text``, or ``None``.

        The request budget measures the WHOLE request with a real tokenizer
        when the endpoint exposes one. A provider that cannot tokenize its
        model's input honestly returns ``None`` (the base contract) and the
        budget falls back to a labelled conservative estimate — it never
        presents an estimate as a tokenizer count.
        """
        return None


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
    logged, or included in ``describe()``.

    Two settings are read from the environment here, both because an
    endpoint's own defaults are not ours to assume.

    ``OR_WM_NO_RESPONSE_FORMAT``: some endpoints reject
    ``response_format`` outright, and an unsupported field must be
    removable without editing code in the field. It is a subtractive
    opt-out — the key is sent unless the variable is set truthy — so an
    unconfigured environment keeps the previous behaviour exactly.

    ``OR_WM_ENABLE_THINKING``: thinking is OFF by default. Measured on
    one endpoint (Qwen3.8-27B via llmapi.paratera.com/v1, same request):
    36.3s / 1291 reasoning tokens by default vs 4.5s / 0 with the key
    sent, and ``reasoning_tokens`` is counted INSIDE
    ``completion_tokens`` (325 total of which 291 were reasoning), so
    thinking competes with the answer for ``max_tokens``. A truthy value
    OMITS the key and lets the endpoint's native default apply;
    ``true`` is never sent, because asserting the opposite default would
    be as much of a guess as assuming it. ``describe()`` records which
    way this instance ran, so the two arms stay distinguishable."""

    name = "http-chat"

    def __init__(self, base_url: str, model: str, api_key: str, *,
                 timeout_s: float = 300.0, max_output_tokens: int = 8192,
                 temperature: float = 0.2):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.max_output_tokens = int(max_output_tokens)
        self.temperature = float(temperature)
        # Whether the JSON-mode hint is sent. Read ONCE at construction so
        # the body built for a call cannot disagree with what ``describe()``
        # reports about that same call.
        self.send_response_format = not _truthy_env(
            "OR_WM_NO_RESPONSE_FORMAT")
        # Whether the model is asked to reason before answering. OFF by
        # default: thinking is charged inside the SAME output budget and
        # was measured to dominate both the token spend and the wall
        # clock. Read once, for the same reason as the field above.
        self.enable_thinking = _truthy_env("OR_WM_ENABLE_THINKING")

    def describe(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "protocol": PROTOCOL_VERSION,
            "model": self.model,
            "base_url": self.base_url,
            "timeout_s": self.timeout_s,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "send_response_format": self.send_response_format,
            "enable_thinking": self.enable_thinking,
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
        — the provider never waits longer than the caller's budget allows.

        NOTE on what that bounds: a socket timeout is the maximum time a
        SINGLE blocking socket operation may stall, not a hard deadline on
        the whole request. A response that trickles bytes for longer than
        the timeout is not interrupted by it, so this bounds the wait per
        operation and the caller's post-call clock check is what bounds
        the call in aggregate. ``describe()`` and the returned
        ``diagnostics.effective`` report the value actually in force.
        """
        effective_timeout = self.timeout_s
        if timeout_s is not None:
            effective_timeout = max(0.001, min(float(timeout_s),
                                               self.timeout_s))
        # Archived predictions are inspected without invoking a provider.
        from or_harness.world_model.strategy_prediction import (
            STRATEGY_OUTCOME_PROTOCOL_VERSION,
            STRATEGY_OUTCOME_SYSTEM_PROMPT,
        )
        from or_harness.world_model.request_budget import (
            strip_request_diagnostics,
        )
        # The budget diagnostics are LOCAL: never serialize them onto the
        # wire, whatever caller assembled the request.
        request = strip_request_diagnostics(request)
        protocol = request.get("prediction_protocol")
        if protocol != STRATEGY_OUTCOME_PROTOCOL_VERSION:
            raise ValueError("unsupported prediction protocol: only prediction_protocol=wm-so/1 is supported")
        prompt = STRATEGY_OUTCOME_SYSTEM_PROMPT
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(
                    request, ensure_ascii=False, default=str)},
            ],
            # JSON mode is ON by default and OMITTED (never null) when the
            # deployment disables it: an endpoint that rejects the field
            # must not receive ``"response_format": null`` in its place.
            **({"response_format": {"type": "json_object"}}
               if self.send_response_format else {}),
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }
        if not self.enable_thinking:
            # Ask the endpoint NOT to reason before answering. vLLM serving
            # Qwen3 reads the flag from ``chat_template_kwargs``: a
            # top-level ``enable_thinking`` is a no-op there, so the model
            # still emits reasoning tokens, and on a large strategy-outcome
            # prompt (~8.5k) they consume the whole ``max_tokens`` — the
            # reply ends with ``finish_reason="length"`` and
            # ``content=null``, which every prediction reads as "model
            # returned no content". Send BOTH keys: the
            # ``chat_template_kwargs`` form for vLLM(Qwen3), and the
            # top-level form which other endpoints read. With the switch on
            # NEITHER key is sent, so the endpoint's own default governs and
            # no vendor default is hard-coded on our side.
            body["enable_thinking"] = False
            body["chat_template_kwargs"] = {"enable_thinking": False}
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
        first = choices[0]
        content = first.get("message", {}).get("content")
        # ``finish_reason`` is the AUTHORITATIVE evidence of how the answer
        # ended. A MISSING field is recorded as ``None`` — never assumed to
        # mean the model stopped on purpose.
        finish_reason = first.get("finish_reason")
        usage = envelope.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = details.get("reasoning_tokens")
        payload = None
        parse_error = None
        if isinstance(content, str):
            if content.strip():
                cleaned = _strip_code_fence(content)
                try:
                    payload = json.loads(cleaned)
                except json.JSONDecodeError as exc:
                    parse_error = f"model content is not valid JSON: {exc}"
            else:
                parse_error = ("model content is empty: the endpoint "
                               "returned no answer text")
        elif content is None:
            parse_error = "model returned no content"
        else:
            # Some endpoints return a structured content list rather than a
            # string. Reported honestly instead of being coerced.
            parse_error = (f"model content has unexpected type "
                           f"{type(content).__name__}")
        raw_content_chars = len(content) if isinstance(content, str) else None
        if content is None:
            content_kind = "null"
        elif isinstance(content, str):
            content_kind = "empty" if not content.strip() else "text"
        else:
            content_kind = type(content).__name__
        return {
            "payload": payload,
            # The FULL usage object is passed through: the token口径
            # (prompt + completion, with reasoning/cached as sub-facts) is
            # decided ONCE in ``world_model.usage``, not here. Keeping
            # ``prompt_tokens`` in the block is what closed the "completion
            # only" gap — it was parsed and then discarded before.
            "usage": dict(usage) if usage else None,
            "error": parse_error,
            "latency_s": latency,
            # Everything a reader needs to tell WHY a call produced no
            # usable answer, without re-running it or reading today's
            # config. Counts only — the reasoning TEXT is never retained.
            "diagnostics": {
                "finish_reason": finish_reason,
                "content_kind": content_kind,
                "raw_content_chars": raw_content_chars,
                "reasoning_tokens": reasoning_tokens,
                "effective": {
                    "max_tokens": self.max_output_tokens,
                    "temperature": self.temperature,
                    "timeout_s": effective_timeout,
                    "provider_timeout_s": self.timeout_s,
                    "model": self.model,
                },
            },
        }
