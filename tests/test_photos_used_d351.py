"""D351: several photos of one ECG — the page gathers them one by one, and the conclusion says how many were read."""
from __future__ import annotations

from pathlib import Path

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer1_ingestion.digitize import render_ecg_page


def test_two_photos_of_the_same_leads_say_one_of_two_was_read(tmp_path):
    from cardiolens.pipeline import run_pipeline
    from cardiolens.telegram.analyze import PHOTOS_USED_NOTE, digitize_photos

    a = render_ecg_page(generate_synthetic_ecg(heart_rate_bpm=72, seed=5), tmp_path / "a.png", px_per_mm=10)
    b = render_ecg_page(generate_synthetic_ecg(heart_rate_bpm=90, seed=6), tmp_path / "b.png", px_per_mm=10)
    sig = digitize_photos([a, b])
    assert f"{PHOTOS_USED_NOTE}1 of 2" in sig.provenance.notes
    res = run_pipeline(sig, mode="A")
    f = next(x for x in res.findings.findings if x.code == "photos_not_all_used")
    assert "Разобрано 1 фото из 2" in f.label_ru and "повторяют уже прочитанные отведения: 1" in f.label_ru
    assert "2 ta suratdan 1 tasi" in f.label_uz and f.group == "quality"
    # review: a note on how the record was read is not an «implausible» finding (no false «artefact» line)
    assert "photos_not_all_used" not in (res.causal_graph.implausible_flags or [])
    from cardiolens.outputs import format_conclusion
    text = format_conclusion(res, "ru")
    assert "Разобрано 1 фото из 2" in text and "перепутанные электроды" not in text


def test_pages_with_different_leads_count_as_all_read():
    """A combined record that took new leads from every page: all photos counted, no finding."""
    from types import SimpleNamespace as NS

    from cardiolens.telegram.analyze import PHOTOS_USED_NOTE, _note_used

    def page(leads):
        return NS(provenance=NS(lead_windows={x: (0.0, 2.5) for x in leads}, notes=[]), lead_names=list(leads))
    limb, chest = page(["I", "II", "III", "aVR", "aVL", "aVF"]), page(["V1", "V2", "V3", "V4", "V5", "V6"])
    combined = page(limb.lead_names + chest.lead_names)
    _note_used(combined, [limb, chest], 2, unread=[])
    assert f"{PHOTOS_USED_NOTE}2 of 2" in combined.provenance.notes
    one = page(["I", "II"])
    _note_used(one, [one], 3, unread=[])                                             # the first read photo only
    assert f"{PHOTOS_USED_NOTE}1 of 3" in one.provenance.notes
    lost = page(["I", "II"])
    _note_used(lost, [lost], 3, unread=["b.jpg", "c.jpg"])                           # two photos could not be read
    assert f"{PHOTOS_USED_NOTE}1 of 3; unread: 2" in lost.provenance.notes


def test_an_unread_photo_is_said_as_unread_not_as_a_repeat(tmp_path):
    from cardiolens.pipeline import run_pipeline
    from cardiolens.telegram.analyze import digitize_photos

    a = render_ecg_page(generate_synthetic_ecg(heart_rate_bpm=72, seed=5), tmp_path / "a.png", px_per_mm=10)
    bad = tmp_path / "bad.png"
    import cv2
    import numpy as np
    cv2.imwrite(str(bad), np.full((900, 1200, 3), 255, np.uint8))                  # a blank page: no ECG
    res = run_pipeline(digitize_photos([a, bad]), mode="A")
    f = next(x for x in res.findings.findings if x.code == "photos_not_all_used")
    assert "не прочитано: 1" in f.label_ru and "повторяют" not in f.label_ru


def test_one_photo_adds_no_note(tmp_path):
    from cardiolens.telegram.analyze import PHOTOS_USED_NOTE, digitize_photos

    a = render_ecg_page(generate_synthetic_ecg(heart_rate_bpm=72, seed=5), tmp_path / "a.png", px_per_mm=10)
    assert not any(n.startswith(PHOTOS_USED_NOTE) for n in digitize_photos([a]).provenance.notes)


def test_the_page_gathers_photos_instead_of_replacing_them():
    page = (Path(__file__).resolve().parents[1] / "src" / "cardiolens" / "api" / "static" / "index.html").read_text()
    assert "files.addEventListener('change', () => addFiles([...files.files]))" in page
    assert "drop.addEventListener('drop', e => { addFiles(e.dataTransfer.files); })" in page
    assert "addFiles(cam.shots)" in page and "kept.some(k => sameFile(k, f))" in page and "kept.splice(i, 1)" in page
    assert "kept = []; files.value = ''" in page                       # «new patient» starts empty
    assert "if (freshSet) { kept = []; freshSet = false; }" in page and "if (!redraw) freshSet = true;" in page
