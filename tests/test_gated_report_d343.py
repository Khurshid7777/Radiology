"""D343: a record under a reliability gate — no drawn trace in the PDF, a deterministic narrative with a fixed next
step, no confidence line; the conclusion text never prints the model line and the Uzbek part no Russian reasoning."""
import numpy as np

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer6_interpret import RECHECK_RECORD, RETAKE_PAPER, RETAKE_SCREEN
from cardiolens.outputs.conclusion import format_conclusion
from cardiolens.pipeline import run_pipeline


class _Loud:
    """A provider that must not be asked about a gated record."""
    name = "claude"
    model = "test-model"

    def generate(self, data):
        raise AssertionError("the model was asked about a gated record")


def _flat():
    s = generate_synthetic_ecg(heart_rate_bpm=70, duration_s=10.0, seed=1)
    return s.model_copy(update={"signal": np.zeros_like(s.signal)})


def test_gated_record_gets_the_deterministic_narrative_and_fixed_next_step():
    r = run_pipeline(_flat(), mode="A", provider=_Loud())
    assert "no_ecg_signal" in r.findings.codes()
    assert r.report.provider == "stub"
    assert (r.report.next_step_ru, r.report.next_step_uz) == RECHECK_RECORD
    ru, uz = format_conclusion(r, "ru"), format_conclusion(r, "uz")
    assert "Уверенность" not in ru and "Ishonch" not in uz
    assert "narrative by" not in ru + uz and "_" not in ru.split("РАССУЖДЕНИЕ")[1].split("Инструмент")[0]


def test_photo_next_steps_differ_for_paper_and_screen():
    assert "лист" in RETAKE_PAPER[0] and "экран" not in RETAKE_PAPER[0]
    assert "экран" in RETAKE_SCREEN[0] and "varaq" in RETAKE_PAPER[1] and "ekran" in RETAKE_SCREEN[1]


def test_model_line_and_foreign_reasoning_stay_out_of_the_text():
    r = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=70, duration_s=10.0, seed=1), mode="A")
    r.report.provider = "claude"
    r.report.reasoning_trace = ["narrative by Claude test-model", "ритм синусовый, интервалы в норме"]
    ru, uz = format_conclusion(r, "ru"), format_conclusion(r, "uz")
    assert "narrative by" not in ru and "ритм синусовый, интервалы в норме" in ru
    assert "ритм синусовый" not in uz and "til modeli" in uz
    assert "Уверенность" in ru                       # a record that is not gated keeps its confidence line


def test_pdf_draws_no_trace_for_a_gated_record(tmp_path, monkeypatch):
    import cardiolens.outputs.pdf_report as P

    seen = []
    real = P.Image
    monkeypatch.setattr(P, "Image", lambda path, **kw: (seen.append(str(path)), real(path, **kw))[1])
    P.write_pdf(run_pipeline(_flat(), mode="A"), str(tmp_path / "g.pdf"))
    assert not any(p.endswith(("wave.png", "graph.png")) for p in seen)
    seen.clear()
    P.write_pdf(run_pipeline(generate_synthetic_ecg(heart_rate_bpm=70, duration_s=10.0, seed=1), mode="A"),
                str(tmp_path / "ok.pdf"))
    assert any(p.endswith("wave.png") for p in seen)
