"""D260: the bot's verdict (comment, source) is kept; the summary counts verdicts without the reviewer."""

import json

from fastapi.testclient import TestClient

from cardiolens.api.app import app
from cardiolens.feedback import Feedback, FeedbackStore, summarize_feedback
from cardiolens.feedback.store import COMMENT_MAX

client = TestClient(app, headers={"X-API-Key": "test-key"})


def test_comment_is_capped_and_long_digits_masked():
    fb = Feedback(analysis_id="a", comment="ФП, не синус. тел 998901234567 " + "x" * 400, source="bot")
    assert "998901234567" not in fb.comment and "…" in fb.comment
    assert len(fb.comment) == COMMENT_MAX
    assert Feedback(analysis_id="a", comment="QRS 120").comment == "QRS 120"   # short numbers stay
    assert Feedback(analysis_id="a", comment="   ").comment is None


def test_old_log_lines_still_read(tmp_path):
    p = tmp_path / "fb.jsonl"
    p.write_text(json.dumps({"analysis_id": "old", "overall_agreement": True}) + "\n")
    fb = FeedbackStore.at(p).all()[0]
    assert fb.comment is None and fb.source is None


def test_summary_counts_by_source_without_reviewer():
    items = [Feedback(analysis_id="1", reviewer="tg:111", overall_agreement=True, source="bot"),
             Feedback(analysis_id="2", reviewer="tg:222", overall_agreement=False, source="bot-group",
                      comment="трепетание 2:1"),
             Feedback(analysis_id="3", overall_agreement=None)]
    s = summarize_feedback(items)
    assert (s["n"], s["agree"], s["disagree"], s["unrated"]) == (3, 1, 1, 1)
    assert s["by_source"]["bot-group"]["disagree"] == 1 and s["by_source"]["web"]["unrated"] == 1
    assert s["disagreed"] == [{"analysis_id": "2", "source": "bot-group", "comment": "трепетание 2:1"}]
    assert "tg:" not in json.dumps(s)
    assert summarize_feedback(items, since_ts=items[0].timestamp + 1e6)["n"] == 0


def test_api_keeps_comment_and_reports_it(tmp_path, monkeypatch):
    monkeypatch.setenv("CARDIOLENS_FEEDBACK_PATH", str(tmp_path / "fb.jsonl"))
    aid = client.post("/api/analyze", data={"case": "stemi", "mode": "A"}).json()["analysis_id"]
    r = client.post(f"/api/analysis/{aid}/feedback",
                    json={"analysis_id": aid, "reviewer": "tg:5", "overall_agreement": False,
                          "findings": [], "comment": "на самом деле перикардит", "source": "bot"})
    assert r.status_code == 200
    saved = FeedbackStore.at(tmp_path / "fb.jsonl").all()[0]
    assert saved.comment == "на самом деле перикардит" and saved.source == "bot"
    fb = client.get("/api/quality/summary?days=7").json()["feedback"]
    assert fb["disagree"] == 1 and fb["disagreed"][0]["comment"] == "на самом деле перикардит"
    assert "tg:5" not in json.dumps(fb)


def test_agent_key_gets_no_comments(tmp_path, monkeypatch):
    monkeypatch.setenv("ECGSCAN_API_KEYS", "tester:test-key,agents:agent-key")
    monkeypatch.setenv("CARDIOLENS_FEEDBACK_PATH", str(tmp_path / "fb.jsonl"))
    FeedbackStore.at(tmp_path / "fb.jsonl").append(
        Feedback(analysis_id="x1", overall_agreement=False, comment="Иванов, ФП", source="bot"))
    agent = TestClient(app, headers={"X-API-Key": "agent-key"})
    fb = agent.get("/api/quality/summary?days=7").json()["feedback"]
    assert fb["disagree"] == 1 and fb["comments_withheld"] is True
    assert fb["disagreed"] == [{"analysis_id": "x1", "source": "bot", "comment": None}]
    assert "Иванов" not in json.dumps(fb)
