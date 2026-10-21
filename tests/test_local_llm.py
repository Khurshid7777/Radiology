"""Local OpenAI-compatible LLM provider (TZ §12, §17): mocked server, fallback, selection."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from cardiolens.config import LLMConfig, load_app_config
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer6_interpret import (
    LocalOpenAIProvider,
    StubProvider,
    default_provider,
    local_available,
    provider_status,
)
from cardiolens.pipeline import run_pipeline


class _Handler(BaseHTTPRequestHandler):
    """Answers /models and /chat/completions with a canned grounded/ungrounded mix."""

    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        self.send_response(200 if self.path.endswith("/models") else 404)
        self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(b'{"data": []}')

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n))
        self.server.last_request = req
        content = json.dumps({
            "summary": [{"text_ru": "Синусовый ритм", "text_uz": "Sinus ritmi", "grounded_by": ["sinus_rhythm"]},
                        {"text_ru": "Выдумка", "text_uz": "Uydirma", "grounded_by": ["not_a_real_token"]}],
            "causal_explanation": [], "differential": [], "red_flags": [],
            "confidence_word_ru": "высокая", "confidence_word_uz": "yuqori",
            "next_step_ru": "Наблюдение", "next_step_uz": "Kuzatuv", "reasoning_trace": ["mock"],
        }, ensure_ascii=False)
        body = json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def mock_server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    yield f"http://127.0.0.1:{srv.server_port}/v1", srv
    srv.shutdown()


def test_local_provider_end_to_end_with_grounding(mock_server):
    url, srv = mock_server
    provider = LocalOpenAIProvider(url, model="mock-model", timeout_s=5)
    r = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=75, seed=9), mode="A", provider=provider)
    texts = [c.text_ru for c in r.report.summary]
    assert "Синусовый ритм" in texts
    assert "Выдумка" not in texts                      # ungrounded claim dropped by the validator
    assert any("local model mock-model" in t for t in r.report.reasoning_trace)
    assert srv.last_request["model"] == "mock-model"
    assert "grounding_tokens" in srv.last_request["messages"][1]["content"]


def test_local_provider_falls_back_to_stub_when_unreachable():
    provider = LocalOpenAIProvider("http://127.0.0.1:9/v1", timeout_s=1)
    r = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=75, seed=9), mode="A", provider=provider)
    assert r.report.summary                            # deterministic narrative still produced
    assert any("local LLM unavailable" in t for t in r.report.reasoning_trace)
    assert local_available("http://127.0.0.1:9/v1") is False


def test_provider_selection_from_config(mock_server, monkeypatch):
    url, _ = mock_server
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(default_provider(llm=LLMConfig(provider="stub")), StubProvider)
    assert isinstance(default_provider(llm=LLMConfig(provider="local", local_base_url=url)), LocalOpenAIProvider)
    assert isinstance(default_provider(llm=LLMConfig(provider="auto", local_base_url=url)), LocalOpenAIProvider)
    assert isinstance(default_provider(llm=LLMConfig(provider="auto", local_base_url="http://127.0.0.1:9/v1")), StubProvider)
    assert load_app_config().llm.provider == "auto"


def test_report_names_its_provider(mock_server):
    url, _ = mock_server
    sig = generate_synthetic_ecg(heart_rate_bpm=75, seed=9)
    live = run_pipeline(sig, mode="A", provider=LocalOpenAIProvider(url, model="mock-model", timeout_s=5))
    assert (live.report.provider, live.report.provider_model) == ("local", "mock-model")
    down = run_pipeline(sig, mode="A", provider=LocalOpenAIProvider("http://127.0.0.1:9/v1", timeout_s=1))
    assert down.report.provider == "stub" and down.report.provider_model is None   # fallback never claims a model
    assert run_pipeline(sig, mode="A", provider=StubProvider()).report.provider == "stub"


def test_provider_status_reports_what_default_provider_resolves_to(mock_server, monkeypatch):
    url, _ = mock_server
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    up = provider_status(LLMConfig(provider="auto", local_base_url=url, local_model="mock-model"))
    assert up["active"] == "local" and up["model"] == "mock-model" and up["local_server_up"] is True
    down = provider_status(LLMConfig(provider="auto", local_base_url="http://127.0.0.1:9/v1"))
    assert down["active"] == "stub" and down["model"] is None and down["local_server_up"] is False
    forced = provider_status(LLMConfig(provider="local", local_base_url="http://127.0.0.1:9/v1"))
    assert forced["configured"] == "local" and forced["active"] == "local"   # configured, will fall back per report


@pytest.fixture
def env_llm(mock_server, monkeypatch):
    """Point app.yaml's llm block at the mock server through the environment (docker-compose path)."""
    url, srv = mock_server
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CARDIOLENS_LLM_PROVIDER", "auto")
    monkeypatch.setenv("CARDIOLENS_LLM_BASE_URL", url)
    monkeypatch.setenv("CARDIOLENS_LLM_MODEL", "mock-model")
    monkeypatch.setenv("CARDIOLENS_LLM_TIMEOUT_S", "5")
    load_app_config.cache_clear()
    yield url, srv
    load_app_config.cache_clear()


def test_env_overrides_llm_config(env_llm):
    url, _ = env_llm
    llm = load_app_config().llm
    assert (llm.provider, llm.local_base_url, llm.local_model, llm.timeout_s) == ("auto", url, "mock-model", 5.0)


def test_web_service_uses_the_local_model_and_says_so(env_llm):
    from fastapi.testclient import TestClient

    from cardiolens.api.app import app

    url, srv = env_llm
    client = TestClient(app, headers={"X-API-Key": "test-key"})
    h = client.get("/health").json()
    assert h["status"] == "ok"
    assert h["llm"]["active"] == "local" and h["llm"]["model"] == "mock-model"
    assert h["llm"]["local_base_url"] == url and h["llm"]["local_server_up"] is True

    d = client.post("/api/analyze", data={"case": "stemi", "mode": "A"}).json()
    it = d["interpretation"]
    assert it["provider"] == "local" and it["provider_model"] == "mock-model"
    assert any("local model mock-model" in t for t in it["reasoning_trace"])
    assert srv.last_request["model"] == "mock-model"
    assert any(a["code"] == "st_elevation" for a in d["critical_alerts"])   # safety layer independent of the LLM
