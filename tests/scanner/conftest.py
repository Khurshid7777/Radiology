import numpy as np
import pytest

from ecgscan.contract import Calibration, Layout, QualitySummary, ScanResult, Source, Trace


@pytest.fixture(scope="session")
def synth_mv():
    """10-s synthetic 12-lead ECG as {lead: mV} at 500 Hz."""
    from cardiolens.layer1_ingestion import generate_synthetic_ecg

    s = generate_synthetic_ecg(heart_rate_bpm=72, seed=7)
    return {n: s.lead(n).astype(np.float64) / 1000.0 for n in s.lead_names}


def make_result(**over) -> ScanResult:
    fs = 500.0
    base = dict(
        scan_id="00000000-0000-0000-0000-000000000000", status="ok",
        source=Source(original_sha256="a" * 64),
        calibration=Calibration(paper_speed_mm_s=25, gain_mm_mv=10, speed_source="user", gain_source="user",
                                confidence=1.0),
        layout=Layout(name="3x4+rhythm", confidence=1.0), sampling_rate_hz=fs,
        leads={"I": Trace.from_array(np.sin(np.arange(1250) / 50), start_s=0.0, fs=fs, quality=1.0)},
        quality=QualitySummary(overall=0.9),
    )
    base.update(over)
    return ScanResult(**base)


@pytest.fixture
def result_factory():
    return make_result
