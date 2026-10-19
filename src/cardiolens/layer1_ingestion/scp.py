"""SCP-ECG (EN 1064 / ISO 11073-91064) rhythm data to the unified representation (D95).

Record layout: CRC-CCITT (2 bytes, over the rest of the record) and record length (4),
then sections, each with a 16-byte header (CRC, id, length incl. header, versions,
"SCPECG"). Section 0 points to the others (id, length, 1-based byte index in the
record). Read here:

* section 1 — only age (tag 4) and sex (tag 8); names, IDs and dates stay in the file;
* section 2 — Huffman tables: absent / 0 tables = plain 16-bit samples, 19999 = the
  standard's default table;
* section 3 — leads (start/end sample, lead id) and flags;
* section 6 — rhythm data: AVM (nV per unit), sample interval (us), difference coding
  (0/1/2), bimodal flag, per-lead byte counts, data.

Checked against the C# ECG Toolkit (``SCPSection2.InhouseDecode``): the default table
codes n ones, a 0, then a sign bit for |value| = n (1..8); 0 is a single 0 bit; nine
ones then 0/1 escape to an 8/16-bit two's-complement original value.

Refused, never guessed (TZ §0): a bad CRC, reference-beat subtraction (section 3 flag),
bimodal compression and custom Huffman tables — each needs a real device file to
verify against before it is supported.
"""

from __future__ import annotations

import binascii
import struct
from pathlib import Path

import numpy as np

from cardiolens.contracts.signal import Patient, Provenance, SignalRepresentation

SCP_LEADS = {1: "I", 2: "II", 3: "V1", 4: "V2", 5: "V3", 6: "V4", 7: "V5", 8: "V6",
             61: "III", 62: "aVR", 63: "aVL", 64: "aVF"}
DEFAULT_HUFFMAN = 19999
_AGE_UNITS = {1: 1.0, 2: 1 / 12, 3: 7 / 365.25, 4: 1 / 365.25, 5: 1 / 8766}


class _Bits:
    def __init__(self, data: bytes):
        self.data, self.pos, self.end = data, 0, len(data) * 8

    def bit(self) -> int:
        if self.pos >= self.end:
            raise EOFError
        b = (self.data[self.pos >> 3] >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return b

    def bits(self, n: int) -> int:
        v = 0
        for _ in range(n):
            v = (v << 1) | self.bit()
        return v


def decode_default_huffman(data: bytes, n: int) -> np.ndarray:
    """``n`` values coded with the SCP-ECG default Huffman table (before difference decoding)."""
    out = np.zeros(n, np.int64)
    r = _Bits(data)
    try:
        for i in range(n):
            ones = 0
            while ones < 9 and r.bit():
                ones += 1
            if ones == 0:
                v = 0
            elif ones < 9:
                v = -ones if r.bit() else ones
            else:
                width = 16 if r.bit() else 8
                v = r.bits(width)
                if v >= 1 << (width - 1):
                    v -= 1 << width
            out[i] = v
    except EOFError:
        raise ValueError(f"SCP-ECG Huffman data ends after {i} of {n} samples") from None
    return out


def _undifference(v: np.ndarray, mode: int) -> np.ndarray:
    if mode == 0:
        return v
    out = np.empty_like(v)
    for i in range(len(v)):
        if mode == 1:
            out[i] = v[i] if i == 0 else v[i] + out[i - 1]
        else:
            out[i] = v[i] if i < 2 else v[i] + 2 * out[i - 1] - out[i - 2]
    return out


def _sections(buf: bytes) -> dict[int, bytes]:
    if len(buf) < 6 + 16:
        raise ValueError("not an SCP-ECG record (too short)")
    crc, length = struct.unpack_from("<HI", buf, 0)
    if length > len(buf) or length < 22:
        raise ValueError("not an SCP-ECG record (length field)")
    if binascii.crc_hqx(buf[2:length], 0xFFFF) != crc:
        raise ValueError("SCP-ECG record CRC mismatch: file corrupt or not SCP-ECG")
    s0_len = struct.unpack_from("<I", buf, 6 + 4)[0]
    if buf[6 + 10:6 + 16] != b"SCPECG" and buf[6 + 10:6 + 16] != b"\x00" * 6:
        raise ValueError("not an SCP-ECG record (section 0 header)")
    out = {}
    for p in range(6 + 16, 6 + s0_len, 10):
        sid, slen, idx = struct.unpack_from("<HII", buf, p)
        if slen > 16 and idx > 0:
            out[sid] = buf[idx - 1 + 16:idx - 1 + slen]       # section body, header skipped
    return out


def _patient(s1: bytes | None) -> Patient:
    age = sex = None
    p = 0
    while s1 and p + 3 <= len(s1):
        tag, n = s1[p], struct.unpack_from("<H", s1, p + 1)[0]
        val = s1[p + 3:p + 3 + n]
        if tag == 255:
            break
        if tag == 4 and n == 3:
            num, unit = struct.unpack_from("<HB", val, 0)
            age = num * _AGE_UNITS[unit] if unit in _AGE_UNITS and num else None
        elif tag == 8 and n == 1:
            sex = {1: "male", 2: "female"}.get(val[0])
        p += 3 + n
    return Patient(age_years=age, sex=sex)


def is_scp(content: bytes) -> bool:
    """An SCP-ECG record: the record CRC holds and section 0 carries the ``SCPECG`` mark."""
    if len(content) < 22 or content[16:22] != b"SCPECG":
        return False
    crc, length = struct.unpack_from("<HI", content, 0)
    return 22 <= length <= len(content) and binascii.crc_hqx(content[2:length], 0xFFFF) == crc


def load_scp(path: str | Path) -> SignalRepresentation:
    """SCP-ECG rhythm data (section 6) in microvolts; see the module docstring for scope."""
    buf = Path(path).read_bytes()
    sec = _sections(buf)
    if 3 not in sec or 6 not in sec:
        raise ValueError("SCP-ECG record without lead definitions (3) or rhythm data (6)")
    s3 = sec[3]
    n_leads, flags = s3[0], s3[1]
    if flags & 0x1:
        raise ValueError("SCP-ECG reference-beat subtraction is not supported yet (needs a sample file)")
    leads = [struct.unpack_from("<IIB", s3, 2 + 9 * i) for i in range(n_leads)]
    tables = struct.unpack_from("<H", sec[2], 0)[0] if 2 in sec and len(sec[2]) >= 2 else 0
    if tables not in (0, DEFAULT_HUFFMAN):
        raise ValueError("SCP-ECG custom Huffman tables are not supported yet (needs a sample file)")
    s6 = sec[6]
    avm_nv, interval_us, diff, bimodal = struct.unpack_from("<HHBB", s6, 0)
    if bimodal:
        raise ValueError("SCP-ECG bimodal compression is not supported yet (needs a sample file)")
    if not avm_nv or not interval_us:
        raise ValueError("SCP-ECG rhythm data without amplitude or time scale")
    if diff not in (0, 1, 2):
        raise ValueError(f"SCP-ECG difference coding {diff} is undefined")
    sizes = struct.unpack_from(f"<{n_leads}H", s6, 6)
    p = 6 + 2 * n_leads
    names, rows = [], []
    for (start, end, lead_id), size in zip(leads, sizes, strict=True):
        n = end - start + 1
        data = s6[p:p + size]
        p += size
        if tables == DEFAULT_HUFFMAN:
            v = decode_default_huffman(data, n)
        else:
            if size < 2 * n:
                raise ValueError("SCP-ECG plain rhythm data shorter than its lead length")
            v = np.frombuffer(data[:2 * n], dtype="<i2").astype(np.int64)
        rows.append(_undifference(v, diff) * (avm_nv / 1000.0))   # nV per unit -> uV
        names.append(SCP_LEADS.get(lead_id, f"scp_lead_{lead_id}"))
    n_min = min(len(r) for r in rows)
    fs = 1e6 / interval_us
    coding = "default Huffman" if tables else "uncompressed"
    return SignalRepresentation(
        signal=np.vstack([r[:n_min] for r in rows]).astype(np.float32),
        sampling_rate_hz=fs,
        lead_names=names,
        duration_s=float(n_min / fs),
        source="scp",
        patient=_patient(sec.get(1)),
        provenance=Provenance(digitized=False, confidence=1.0,
                              notes=[f"SCP-ECG, {n_leads} leads, difference {diff}, {coding}"]),
    )
