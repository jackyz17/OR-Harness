"""Provider-independent LLM interface with test and command adapters."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence


class LLMClient(Protocol):
    async def generate_text(self, prompt: str, timeout: Optional[float] = None) -> str:
        ...

    async def generate_object(self, prompt: str, timeout: Optional[float] = None) -> Any:
        ...


class LLMClientError(RuntimeError):
    pass


class FakeLLMClient:
    """Injectable deterministic responses for tests only."""

    def __init__(self, text_responses: Optional[Sequence[str]] = None, object_responses=None):
        self.text_responses = list(text_responses or [])
        self.object_responses = list(object_responses or [])
        self.prompts: List[str] = []

    async def generate_text(self, prompt: str, timeout: Optional[float] = None) -> str:
        self.prompts.append(prompt)
        if not self.text_responses:
            return "# no generated code"
        return self.text_responses.pop(0)

    async def generate_object(self, prompt: str, timeout: Optional[float] = None) -> Any:
        self.prompts.append(prompt)
        if not self.object_responses:
            return []
        value = self.object_responses.pop(0)
        if isinstance(value, str):
            return json.loads(value)
        return value


class CommandLLMClient:
    """Adapt an existing Hermes/provider wrapper command using JSON stdin/stdout.

    The fixed argv is configuration, never derived from user problem text.
    """

    def __init__(self, command: Sequence[str], timeout_seconds: float = 120):
        if not command:
            raise ValueError("LLM command cannot be empty")
        self.command = list(command)
        self.timeout_seconds = timeout_seconds

    async def _call(self, prompt: str, response_format: str, timeout: Optional[float]) -> Any:
        process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        payload = json.dumps({"prompt": prompt, "response_format": response_format}).encode("utf-8")
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(payload), timeout=timeout or self.timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.wait()
            raise LLMClientError("LLM command timed out") from exc
        if process.returncode != 0:
            raise LLMClientError("LLM command failed: " + stderr.decode("utf-8", "replace")[-1000:])
        text = stdout.decode("utf-8", "replace").strip()
        return json.loads(text) if response_format == "json" else text

    async def generate_text(self, prompt: str, timeout: Optional[float] = None) -> str:
        return str(await self._call(prompt, "text", timeout))

    async def generate_object(self, prompt: str, timeout: Optional[float] = None) -> Any:
        return await self._call(prompt, "json", timeout)


class StdinLLMClient:
    """Harness-mode LLM client: the harness agent IS the LLM (D18).

    Used with CLI --interactive-llm. The framework prints each prompt to stderr;
    the harness agent reads it, generates the answer with its own reasoning, and
    feeds it back on stdin, terminated by a single line with the end marker.
    No wrapper script, no external API, no credentials — the agent answers directly.
    """

    def __init__(self, end_marker: str = "<<<END_LLM>>>"):
        if not end_marker:
            raise ValueError("end marker cannot be empty")
        self.end_marker = end_marker

    async def _ask(self, prompt: str, response_format: str) -> str:
        loop = asyncio.get_running_loop()
        header = "===== LLM PROMPT (respond {}, end with a line '{}') =====".format(
            response_format, self.end_marker
        )
        sys.stderr.write("\n" + header + "\n" + prompt + "\n" + "=" * len(header) + "\n")
        sys.stderr.flush()
        lines: List[str] = []
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if line == "":  # EOF (readline returns '' at end of input, not None)
                break
            stripped = line.rstrip("\n").rstrip("\r")
            if stripped == self.end_marker:
                break
            lines.append(line)
        return "\n".join(lines).strip()

    async def generate_text(self, prompt: str, timeout: Optional[float] = None) -> str:
        return await self._ask(prompt, "text")

    async def generate_object(self, prompt: str, timeout: Optional[float] = None) -> Any:
        raw = await self._ask(prompt, "json")
        if not raw:
            return []
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return raw

