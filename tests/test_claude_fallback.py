"""Claude narrative provider (TZ §12, D154): a failed call falls back to the offline narrative.

The ``anthropic`` SDK is replaced by a fake module, so the tests run without the
package, a key or the network — what matters is what the provider does with a
reply or a failure, not the SDK.
"""

import json
import sys
import types

import pytest

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer6_interpret import ClaudeProvider
from cardiolens.pipeline import run_pipeline

_REPLY = json.dumps({
    "summary": [{"text_ru": "Синусовый ритм", "text_uz": "Sinus ritmi", "grounded_by": ["sinus_rhythm"]},
                {"text_ru": "Выдумка", "text_uz": "Uydirma", "grounded_by": ["not_a_real_token"]}],
    "causal_explanation": [], "differential": [], "red_flags": [],
    "confidence_word_ru": "высокая", "confidence_word_uz": "yuqori",
    "next_step_ru": "Наблюдение", "next_step_uz": "Kuzatuv", "reasoning_trace": ["mock"],
}, ensure_ascii=False)


def _fake_anthropic(monkeypatch, *, fail: Exception | None):
    """Install a stand-in ``anthropic`` whose ``messages.create`` answers or raises."""
    mod = types.ModuleType("anthropic")

    class AuthenticationError(Exception):
        pass

    class _Messages:
        def create(self, **kw):
            mod.last_request = kw
            if fail is not None:
                raise fail
            return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=_REPLY)])

    class Anthropic:
        def __init__(self):
            self.messages = _Messages()

    mod.AuthenticationError, mod.Anthropic = AuthenticationError, Anthropic
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return mod


@pytest.fixture
def ecg():
    return generate_synthetic_ecg(heart_rate_bpm=75, seed=9)


def test_claude_reply_is_grounded_and_named(monkeypatch, ecg):
    mod = _fake_anthropic(monkeypatch, fail=None)
    r = run_pipeline(ecg, mode="A", provider=ClaudeProvider(model="test-model"))
    texts = [c.text_ru for c in r.report.summary]
    assert "Синусовый ритм" in texts
    assert "Выдумка" not in texts                      # ungrounded claim dropped by the validator
    assert (r.report.provider, r.report.provider_model) == ("claude", "test-model")
    assert any("narrative by Claude test-model" in t for t in r.report.reasoning_trace)
    assert mod.last_request["model"] == "test-model"
    assert "grounding_tokens" in mod.last_request["messages"][0]["content"]


def test_refused_key_falls_back_to_offline_narrative(monkeypatch, ecg):
    mod = _fake_anthropic(monkeypatch, fail=None)
    mod_fail = _fake_anthropic(monkeypatch, fail=mod.AuthenticationError("invalid x-api-key"))
    assert mod_fail is not mod
    r = run_pipeline(ecg, mode="A", provider=ClaudeProvider(model="test-model"))
    assert r.report.provider == "stub" and r.report.provider_model is None
    assert r.report.summary, "the analysis still carries a narrative"
    assert any("Claude unavailable (AuthenticationError)" in t for t in r.report.reasoning_trace)


def test_malformed_reply_falls_back_to_offline_narrative(monkeypatch, ecg):
    mod = _fake_anthropic(monkeypatch, fail=None)
    mod.Anthropic.__init__ = lambda self: setattr(self, "messages", types.SimpleNamespace(
        create=lambda **kw: types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text="no json here")])))
    r = run_pipeline(ecg, mode="A", provider=ClaudeProvider(model="test-model"))
    assert r.report.provider == "stub"
    assert any("Claude unavailable" in t for t in r.report.reasoning_trace)
