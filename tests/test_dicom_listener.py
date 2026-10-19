"""D96: DICOM C-STORE receiver for ECG devices (real DICOM network protocol on localhost)."""

import json
import socket

import pytest

pytest.importorskip("pynetdicom")

from pynetdicom import AE
from pynetdicom.sop_class import TwelveLeadECGWaveformStorage, Verification

from cardiolens.dicom_listener import EcgListener, ListenerConfig
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from dicom_builder import dataset, group

DEVICE = "TELEECG01"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def ecg_ds():
    sig = generate_synthetic_ecg(heart_rate_bpm=70, seed=11)
    return dataset([group(sig.signal, sig.lead_names, sig.sampling_rate_hz)])


@pytest.fixture
def listener(tmp_path):
    received = []

    def analyze(data: bytes) -> dict:
        from cardiolens.layer1_ingestion import is_dicom

        assert is_dicom(data)
        received.append(data)
        if b"REFUSE" in data:
            raise ValueError("DICOM file is not an ECG waveform")
        return {"analysis_id": "abc123", "heart_rate_bpm": 70.0}

    cfg = ListenerConfig(port=_free_port(), allowed_calling_aes=[DEVICE], log_path=str(tmp_path / "rx.jsonl"))
    lst = EcgListener(cfg, analyze=analyze)
    lst.start()
    yield lst, received
    lst.stop()


def _send(port: int, ds=None, *, calling: str = DEVICE):
    ae = AE(ae_title=calling)
    ae.add_requested_context(TwelveLeadECGWaveformStorage)
    ae.add_requested_context(Verification)
    assoc = ae.associate("127.0.0.1", port, ae_title="CARDIOLENS")
    if not assoc.is_established:
        return None
    try:
        status = assoc.send_c_store(ds) if ds is not None else assoc.send_c_echo()
        return int(status.Status) if status else None
    finally:
        assoc.release()


def test_echo_and_store(listener, ecg_ds):
    lst, received = listener
    assert _send(lst.cfg.port) == 0x0000                      # "test connection" on the device
    assert _send(lst.cfg.port, ecg_ds) == 0x0000
    assert len(received) == 1
    entry = json.loads(lst.log_path.read_text().splitlines()[-1])
    assert entry["status"] == "analysed" and entry["analysis_id"] == "abc123" and entry["device_ae"] == DEVICE


def test_log_has_no_patient_data(listener, ecg_ds):
    lst, _ = listener
    _send(lst.cfg.port, ecg_ds)
    text = lst.log_path.read_text()
    assert "Ivanov" not in text and "MRN-12345" not in text and str(ecg_ds.SOPInstanceUID) not in text


def test_unknown_device_is_not_let_in(listener, ecg_ds):
    lst, received = listener
    assert _send(lst.cfg.port, ecg_ds, calling="STRANGER") is None
    assert received == []


def test_refused_ecg_returns_failure_to_the_device(listener, ecg_ds):
    lst, _ = listener
    bad = dataset(list(ecg_ds.WaveformSequence))
    bad.StudyDescription = "REFUSE"
    assert _send(lst.cfg.port, bad) == 0xC000
    entry = json.loads(lst.log_path.read_text().splitlines()[-1])
    assert entry["status"] == "refused" and "not an ECG" in entry["error"]


def test_no_whitelist_no_listener(tmp_path):
    with pytest.raises(ValueError, match="allowed_calling_aes is empty"):
        EcgListener(ListenerConfig(log_path=str(tmp_path / "x.jsonl")))


def test_end_to_end_through_the_api(tmp_path, ecg_ds):
    """The default analyzer posts to /api/analyze; here the API runs in-process."""
    from fastapi.testclient import TestClient

    from cardiolens.api.app import app

    client = TestClient(app)

    def analyze(data: bytes) -> dict:
        r = client.post("/api/analyze", files=[("files", ("ecg.dcm", data, "application/dicom"))])
        if r.status_code != 200:
            raise ValueError(r.text[:200])
        return {"analysis_id": r.json()["analysis_id"], "heart_rate_bpm": r.json()["measurements"]["heart_rate_bpm"]}

    cfg = ListenerConfig(port=_free_port(), allowed_calling_aes=[DEVICE], log_path=str(tmp_path / "rx.jsonl"))
    lst = EcgListener(cfg, analyze=analyze)
    lst.start()
    try:
        assert _send(cfg.port, ecg_ds) == 0x0000
    finally:
        lst.stop()
    entry = json.loads(lst.log_path.read_text().splitlines()[-1])
    assert entry["heart_rate_bpm"] == pytest.approx(70, abs=3)
    assert client.get(f"/api/analysis/{entry['analysis_id']}/conclusion").status_code == 200
