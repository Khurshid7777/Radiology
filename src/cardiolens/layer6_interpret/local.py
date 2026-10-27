"""Local LLM provider for Layer 6 — any OpenAI-compatible chat endpoint (TZ §12, §17).

Ollama (``ollama serve`` → http://127.0.0.1:11434/v1), llama.cpp server, vLLM,
LM Studio and similar all expose ``POST /chat/completions``. The prompt, the
grounding tokens and the JSON schema are the same as for the cloud provider; the
grounding validator runs afterwards regardless. Nothing leaves the machine.

If the endpoint is unreachable or returns unusable output the provider falls
back to the deterministic ``StubProvider`` and says so in the reasoning trace —
a missing model must never block a report (TZ §12 "ядро офлайн").
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from cardiolens.contracts.report import InterpretationReport
from cardiolens.layer6_interpret.claude import (
    _SYSTEM,
    PROMPT_INSTRUCTIONS,
    ClaudeProvider,
    parse_json_object,
    report_from_json,
)
from cardiolens.layer6_interpret.provider import InterpretationInput
from cardiolens.layer6_interpret.stub import StubProvider

DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"   # Ollama's OpenAI-compatible endpoint
DEFAULT_MODEL = "llama3.1:8b"


def local_available(base_url: str = DEFAULT_BASE_URL, timeout_s: float = 0.5) -> bool:
    """True when an OpenAI-compatible server answers ``GET {base_url}/models``."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=timeout_s) as resp:
            return resp.status == 200
    except Exception:
        return False


class LocalOpenAIProvider(ClaudeProvider):
    """OpenAI-compatible local model; same payload/prompt as the cloud provider."""

    name = "local"

    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                 timeout_s: float = 120.0, max_tokens: int = 2000):
        super().__init__(model=model, max_tokens=max_tokens)
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _request(self, prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": _SYSTEM},
                         {"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": self.max_tokens,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            parsed = json.loads(resp.read().decode("utf-8"))
        return parsed["choices"][0]["message"]["content"]

    def generate(self, data: InterpretationInput) -> InterpretationReport:
        prompt = PROMPT_INSTRUCTIONS + json.dumps(self._payload(data), ensure_ascii=False, indent=2)
        try:
            text = self._request(prompt)
            report = report_from_json(parse_json_object(text), data)
            report.provider, report.provider_model = self.name, self.model
            report.reasoning_trace.insert(0, f"narrative by local model {self.model} at {self.base_url}")
            return report
        except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError) as e:
            report = StubProvider().generate(data)   # provider stays "stub": the UI must not claim a model
            report.reasoning_trace.insert(
                0, f"local LLM unavailable ({type(e).__name__}) -> deterministic offline narrative")
            return report
