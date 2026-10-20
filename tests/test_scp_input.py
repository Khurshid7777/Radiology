"""D95: SCP-ECG rhythm data (default Huffman table or plain samples, difference 0/1/2)."""

import binascii
import struct

import numpy as np
import pytest

from cardiolens.layer1_ingestion import generate_synthetic_ecg, is_scp, load, load_scp
from cardiolens.layer1_ingestion.scp import SCP_LEADS, decode_default_huffman

LEAD_IDS = {v: k for k, v in SCP_LEADS.items()}


def test_default_huffman_hand_built_vector():
    """Codes written out by hand from the standard's table (as the C# ECG Toolkit decodes it)."""
    bits = ("0" + "100" + "101" + "11100" + "1111111101"
            + "111111111" + "0" + "00001001"                   # 9 via the 8-bit escape
            + "111111111" + "1" + format(0xFF38, "016b")       # -200 via the 16-bit escape
            + "111111111" + "1" + format(1000, "016b"))
    bits += "0" * (-len(bits) % 8)
    data = int(bits, 2).to_bytes(len(bits) // 8, "big")
    assert decode_default_huffman(data, 8).tolist() == [0, 1, -1, 3, -8, 9, -200, 1000]


# --- a small SCP-ECG writer for the tests -----------------------------------------------

def _huff(values) -> bytes:
    out = []
    for v in values:
        v = int(v)
        if v == 0:
            out.append("0")
        elif abs(v) <= 8:
            out.append("1" * abs(v) + "0" + ("1" if v < 0 else "0"))
        elif -128 <= v < 128:
            out.append("1" * 9 + "0" + format(v & 0xFF, "08b"))
        else:
            out.append("1" * 9 + "1" + format(v & 0xFFFF, "016b"))
    bits = "".join(out)
    bits += "0" * (-len(bits) % 8)
    return int(bits, 2).to_bytes(len(bits) // 8, "big") if bits else b""


def _difference(v: np.ndarray, mode: int) -> np.ndarray:
    v = v.astype(np.int64)
    if mode == 0:
        return v
    if mode == 1:
        return np.concatenate([v[:1], np.diff(v)])
    d = v.copy()
    d[2:] = v[2:] - 2 * v[1:-1] + v[:-2]
    return d


def _section(sid: int, body: bytes) -> bytes:
    head = struct.pack("<HIBB", sid, 16 + len(body), 20, 20) + b"SCPECG"
    crc = binascii.crc_hqx(head + body, 0xFFFF)
    return struct.pack("<H", crc) + head + body


def _record(units: np.ndarray, names: list[str], *, avm_nv=5000, interval_us=2000, diff=1, huffman=True,
            flags=0x04, bimodal=0, tables=None, age=(52, 1), sex=2, ranges=None, sizes=None) -> bytes:
    n = units.shape[1]
    s1 = (bytes([0]) + struct.pack("<H", 7) + b"Ivanov\0"            # last name
          + bytes([2]) + struct.pack("<H", 6) + b"ID-42\0"           # patient ID
          + bytes([4]) + struct.pack("<HHB", 3, *age)
          + bytes([8]) + struct.pack("<HB", 1, sex) + bytes([255]) + struct.pack("<H", 0))
    ranges = ranges or [(1, n)] * len(names)
    s3 = struct.pack("<BB", len(names), flags) + b"".join(
        struct.pack("<IIB", a, b, LEAD_IDS[nm]) for (a, b), nm in zip(ranges, names, strict=True))
    coded = [(_huff(_difference(u, diff)) if huffman else _difference(u, diff).astype("<i2").tobytes())
             for u in units]
    s6 = struct.pack("<HHBB", avm_nv, interval_us, diff, bimodal) + b"".join(
        struct.pack("<H", k) for k in (sizes or [len(c) for c in coded])) + b"".join(coded)
    body = {1: s1, 3: s3, 6: s6}
    if huffman or tables is not None:
        body[2] = struct.pack("<H", 19999 if tables is None else tables)
    ids = [0, 1, 2, 3, 4, 5, 6, 7]
    sections = {sid: _section(sid, b) for sid, b in body.items()}
    s0_len = 16 + 10 * len(ids)
    pos, ptrs = 6 + s0_len + 1, b""
    for sid in ids:
        if sid in sections:
            ptrs += struct.pack("<HII", sid, len(sections[sid]), pos)
            pos += len(sections[sid])
        else:
            ptrs += struct.pack("<HII", sid, 0, 0)
    rec = _section(0, ptrs) + b"".join(sections[s] for s in ids if s in sections)
    length = 6 + len(rec)
    tail = struct.pack("<I", length) + rec
    return struct.pack("<H", binascii.crc_hqx(tail, 0xFFFF)) + tail


@pytest.fixture(scope="module")
def ecg():
    sig = generate_synthetic_ecg(heart_rate_bpm=66, seed=5)
    units = np.round(sig.signal / 5.0).astype(np.int64)           # AVM 5000 nV = 5 uV per unit
    return sig, units


@pytest.mark.parametrize("diff", [0, 1, 2])
@pytest.mark.parametrize("huffman", [True, False])
def test_rhythm_read_in_microvolts(tmp_path, ecg, diff, huffman):
    sig, units = ecg
    f = tmp_path / "ecg.scp"
    f.write_bytes(_record(units, sig.lead_names, diff=diff, huffman=huffman))
    out = load_scp(f)
    assert out.source == "scp" and out.lead_names == sig.lead_names
    assert out.sampling_rate_hz == 500.0 and out.signal.shape == sig.signal.shape
    assert np.array_equal(out.signal, (units * 5.0).astype(np.float32))
    assert out.patient.age_years == 52 and out.patient.sex == "female"


def test_identity_never_leaves_the_file(tmp_path, ecg):
    sig, units = ecg
    f = tmp_path / "pii.scp"
    f.write_bytes(_record(units, sig.lead_names))
    dumped = load_scp(f).model_dump_json(exclude={"signal"})
    assert "Ivanov" not in dumped and "ID-42" not in dumped


def test_corrupt_or_unsupported_files_are_refused(tmp_path, ecg):
    sig, units = ecg
    rec = bytearray(_record(units, sig.lead_names))
    rec[-3] ^= 0xFF
    (tmp_path / "bad.scp").write_bytes(bytes(rec))
    with pytest.raises(ValueError, match="CRC"):
        load_scp(tmp_path / "bad.scp")
    for kw, msg in (({"flags": 0x05}, "reference-beat"), ({"bimodal": 1}, "bimodal"),
                    ({"tables": 1, "huffman": False}, "custom Huffman")):
        f = tmp_path / "x.scp"
        f.write_bytes(_record(units, sig.lead_names, **kw))
        with pytest.raises(ValueError, match=msg):
            load_scp(f)


def test_dispatch_and_api_without_extension(tmp_path, ecg):
    from fastapi.testclient import TestClient

    from cardiolens.api.app import app

    sig, units = ecg
    data = _record(units, sig.lead_names)
    f = tmp_path / "ecg.scp"
    f.write_bytes(data)
    assert load(f).source == "scp" and is_scp(data) and not is_scp(b"\0" * 64)
    r = TestClient(app, headers={"X-API-Key": "test-key"}).post("/api/analyze", files=[("files", ("ECG0001", data, "application/octet-stream"))])
    assert r.status_code == 200, r.text
    assert r.json()["measurements"]["heart_rate_bpm"] == pytest.approx(66, abs=3)
