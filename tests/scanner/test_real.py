"""M5: real photos with digital originals (TZ §11): loading and one-command report."""

import base64

import numpy as np
import pytest

from ecgscan.bench.real import read_gold, read_muse_xml, real_cases

cv2 = pytest.importorskip("cv2")


def _muse_xml(leads_mv: dict, fs: int = 500) -> str:
    parts = []
    for name, x in leads_mv.items():
        data = base64.b64encode(np.round(x * 1000 / 4.88).astype("<i2").tobytes()).decode()
        parts.append(f"<LeadData><LeadID>{name}</LeadID><LeadAmplitudeUnitsPerBit>4.88</LeadAmplitudeUnitsPerBit>"
                     f"<LeadAmplitudeUnits>MICROVOLTS</LeadAmplitudeUnits><WaveFormData>{data}</WaveFormData>"
                     "</LeadData>")
    return ("<RestingECG><Waveform><WaveformType>Median</WaveformType><SampleBase>500</SampleBase></Waveform>"
            f"<Waveform><WaveformType>Rhythm</WaveformType><SampleBase>{fs}</SampleBase>{''.join(parts)}"
            "</Waveform></RestingECG>")


def test_muse_xml_round_trip_and_derived_limb_leads(tmp_path, synth_mv):
    eight = {k: synth_mv[k] for k in ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")}
    p = tmp_path / "rec.xml"
    p.write_text(_muse_xml(eight))
    raw, fs = read_muse_xml(p)
    assert fs == 500 and set(raw) == set(eight)
    assert np.max(np.abs(raw["II"] - eight["II"])) < 0.003                 # 4.88 µV per bit
    leads, _ = read_gold(p, {})
    assert {"III", "aVR", "aVL", "aVF"} <= set(leads)
    assert np.allclose(leads["III"], leads["II"] - leads["I"])


def test_csv_original_needs_its_rate(tmp_path, synth_mv):
    p = tmp_path / "rec.csv"
    names = list(synth_mv)
    np.savetxt(p, np.stack([synth_mv[n] * 1000 for n in names], axis=1), delimiter=",", header=",".join(names),
               comments="")
    with pytest.raises(ValueError, match="csv.fs"):
        read_gold(p, {})
    leads, fs = read_gold(p, {"csv": {"fs": 500, "unit": "uv"}})
    assert fs == 500 and np.allclose(leads["V3"], synth_mv["V3"], atol=1e-6)


def test_real_folder_to_report(tmp_path, synth_mv):
    from ecgscan.bench.cases import bench_config
    from ecgscan.bench.digitizers import default_digitizers
    from ecgscan.bench.render import RenderSpec, render_page
    from ecgscan.bench.run import run_bench
    from ecgscan.layout import get_layout

    dev = tmp_path / "real" / "device_a"
    dev.mkdir(parents=True)
    (dev / "device.yaml").write_text("layout: 3x4+rhythm\nspeed_mm_s: 25\ngain_mm_mv: 10\n")
    img, _ = render_page(synth_mv, 500.0, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=10.0))
    cv2.imwrite(str(dev / "p001.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (dev / "p001.xml").write_text(_muse_xml({k: synth_mv[k] for k in ("I", "II", "V1", "V2", "V3", "V4", "V5",
                                                                        "V6")}))
    cv2.imwrite(str(dev / "orphan.jpg"), img[:50, :50])                    # no original: skipped
    cases = real_cases(tmp_path / "real")
    assert [c.case_id for c in cases] == ["device_a/p001"] and cases[0].slice == "real_device_a"
    digs = [d for d in default_digitizers(bench_config()) if d.name == "ecgscan"]
    s = run_bench("standard", tmp_path / "rep", digitizers=digs, cases=cases, log=lambda *_: None)
    assert s["acceptance_scope"] == "реальные фото"
    row = s["by_slice"]["ecgscan"]["real_device_a"]
    assert row["r_median"] > 0.9
    assert "реальные фото" in (tmp_path / "rep" / "report.md").read_text(encoding="utf-8")
