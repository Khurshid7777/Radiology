"""R5 (SD-105): phone photos the main segmenter leaves unread go once more through the photo
segmenter; that reading is never "ok" and says so."""


def test_a_page_refused_for_its_layout_is_read_by_the_photo_segmenter_never_ok(monkeypatch, result_factory):
    from ecgscan import pipeline

    calls = []

    def once(image, *, config, **kw):
        calls.append(config["segmentation"]["onnx_model"])
        if len(calls) == 1:
            return result_factory(status="rejected", reject_reasons=["LAYOUT_UNKNOWN"])
        return result_factory(status="ok", warnings=[])

    monkeypatch.setattr(pipeline, "_scan_once", once)
    cfg = pipeline.pipeline_config()
    res = pipeline.scan(b"photo", config=cfg)
    assert calls == [cfg["segmentation"]["onnx_model"], cfg["segmentation"]["fallback"]["onnx_model"]]
    assert res.status == "needs_confirmation" and "SEGMENTER_FALLBACK" in res.warnings


def test_a_page_read_the_first_time_or_refused_for_another_reason_is_left_alone(monkeypatch, result_factory):
    from ecgscan import pipeline

    for first in (result_factory(status="ok"), result_factory(status="rejected", reject_reasons=["NOT_AN_ECG"])):
        calls: list[int] = []
        monkeypatch.setattr(pipeline, "_scan_once",
                            lambda image, *, config, _f=first, _c=calls, **kw: _c.append(1) or _f)
        assert pipeline.scan(b"photo") is first and calls == [1]
