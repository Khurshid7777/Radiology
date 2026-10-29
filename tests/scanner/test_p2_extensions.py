"""P2 (TZ 18.11, SD-097): the extension points exist, accept only what they should, and are off."""

import json

import pytest

from ecgscan.contract.models import Confirmation, LeadRef, ScanResult, Signature
from ecgscan.qr import parse_qr, subject_qr

SIG = dict(method="ed25519", key_id="dr-1", signed_sha256="a" * 64, value="c2lnbmF0dXJl")


@pytest.mark.parametrize("ref", ["sub_8f3k2Lq9", "sub_" + "x" * 64, "sub_a1-b2_c3d4"])
def test_a_pseudonym_is_accepted(result_factory, ref):
    assert result_factory(subject_ref=ref).subject_ref == ref


@pytest.mark.parametrize("ref", ["Каримов Акмал", "1234567", "sub_short", "19.10.1957", "sub_" + "x" * 65,
                                 "SUB_abcdefgh", "sub_abc def gh"])
def test_a_name_a_record_number_or_a_date_is_refused(result_factory, ref):
    with pytest.raises(ValueError):
        result_factory(subject_ref=ref)


def test_the_new_fields_validate_against_the_schema(result_factory):
    jsonschema = pytest.importorskip("jsonschema")
    from ecgscan.contract import SCHEMA_PATH

    v = jsonschema.Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    conf = Confirmation(by="dr", at="2026-10-29T10:00:00Z", decision="accepted", signature=Signature(**SIG))
    res = result_factory(subject_ref="sub_8f3k2Lq9", confirmation=conf)
    v.validate(res.to_json_dict())
    bad = res.to_json_dict()
    bad["subject_ref"] = "Karimov Akmal"
    assert not v.is_valid(bad)
    assert ScanResult.model_validate(res.to_json_dict()) == res


def test_the_content_hash_covers_everything_but_the_signature(result_factory):
    conf = Confirmation(by="dr", at="2026-10-29T10:00:00Z", decision="accepted")
    res = result_factory(confirmation=conf)
    h = res.content_sha256()
    signed = res.model_copy(update={"confirmation": conf.model_copy(update={"signature": Signature(**SIG)})})
    assert signed.content_sha256() == h                                  # signing does not change what is signed
    assert result_factory(confirmation=conf, status="needs_confirmation").content_sha256() != h
    other = res.model_copy(update={"calibration": res.calibration.model_copy(update={"gain_mm_mv": 20})})
    assert other.content_sha256() != h
    assert len(h) == 64 and h == res.content_sha256()                    # stable


def test_a_comment_points_to_an_export_channel_and_a_span():
    assert LeadRef(channel="II_rhythm", t0_s=3.2, t1_s=3.9).channel == "II_rhythm"
    assert LeadRef(channel="V5", t0_s=7.5, t1_s=7.5)
    for bad in (dict(channel="V7", t0_s=0, t1_s=1), dict(channel="II", t0_s=2, t1_s=1),
                dict(channel="II", t0_s=-1, t1_s=1)):
        with pytest.raises(ValueError):
            LeadRef(**bad)


def test_qr_payloads():
    assert subject_qr("sub_8f3k2Lq9") == "cv1:s/sub_8f3k2Lq9"
    p = parse_qr("cv1:s/sub_8f3k2Lq9")
    assert (p.kind, p.value) == ("subject", "sub_8f3k2Lq9")
    assert parse_qr("cv1:l/" + "T" * 40).kind == "login"
    for bad in ("https://evil.example/cv1", "cv1:s/Каримов", "cv1:l/short", "cv2:s/sub_8f3k2Lq9", "cv1:x/abc",
                "cv1:s/1234567"):
        with pytest.raises(ValueError):
            parse_qr(bad)
    with pytest.raises(ValueError):
        subject_qr("Karimov")


def test_every_feature_is_off_and_the_app_is_told(monkeypatch):
    from fastapi.testclient import TestClient

    from ecgscan_service.app import app
    from ecgscan_service.extensions import FEATURES, features
    from ecgscan_service.router import KEYS_ENV

    assert features() == {f: False for f in FEATURES}
    monkeypatch.setenv(KEYS_ENV, "dr:secret-1")
    r = TestClient(app).get("/scans/config/features", headers={"X-API-Key": "secret-1"})
    assert r.status_code == 200 and r.json() == {"features": {f: False for f in FEATURES}}


def test_a_feature_switched_on_without_its_implementation_refuses(monkeypatch):
    from ecgscan_service import extensions

    real = extensions.load_yaml
    monkeypatch.setattr(extensions, "load_yaml", lambda name: {**real(name), "extensions": {"signature": True}})
    monkeypatch.setattr(extensions, "_registry", {})
    with pytest.raises(RuntimeError, match="'signature' is on but \\['signer'\\] not registered"):
        extensions.features()
    extensions.register("signer", object())
    assert extensions.features()["signature"] is True
    with pytest.raises(ValueError, match="unknown extension"):
        extensions.register("payments", object())
    monkeypatch.setattr(extensions, "load_yaml", lambda name: {**real(name), "extensions": {"ads": True}})
    with pytest.raises(ValueError, match="unknown extensions"):
        extensions.switches()
