"""S2 (TZ 18.5): layout templates, the 6x2 / 3x4+3rhythm decision, device profiles, the doctor's layout."""

import pytest

from ecgscan.bench.render import RenderSpec, render_page
from ecgscan.layout import available_layouts, get_layout
from ecgscan.layout.devices import DeviceProfile, _profile, device_profiles
from ecgscan.layout.registry import layout_name
from ecgscan.pipeline import ScanDebug, ScanHints, scan

cv2 = pytest.importorskip("cv2")


def _png(mv, layout: str) -> bytes:
    page, _ = render_page(mv, 500.0, get_layout(layout), RenderSpec(px_per_mm=10.0))
    return cv2.imencode(".png", page)[1].tobytes()


def test_the_v1_templates_are_in_the_registry():
    names = set(available_layouts())
    assert {"3x4+rhythm", "3x4+3rhythm", "6x2", "12x1", "single"} <= names
    assert layout_name("single_strip") == "single" and get_layout("single_strip").name == "single"
    spec = get_layout("3x4+3rhythm")
    assert spec.rhythm == ("V1", "II", "V5") and spec.n_trace_rows == get_layout("6x2").n_trace_rows == 6
    with pytest.raises(KeyError):
        layout_name("4x3")


@pytest.mark.parametrize("layout", ["3x4+3rhythm", "6x2"])
def test_six_rows_are_told_apart_by_the_rhythm_strips_repeats(synth_mv, layout):
    """Same row count: the strips of 3x4+3rhythm repeat their panel cells, 6x2 repeats nothing.
    Before the repeats check a 6x2 page with 3x4+3rhythm in the registry was LAYOUT_UNKNOWN."""
    res = scan(_png(synth_mv, layout))
    assert res.layout.name == layout and res.status == "ok" and "LAYOUT_UNCERTAIN" not in res.warnings
    assert not res.layout.confirmed_by_user and res.layout.device_profile_id is None


def test_the_doctors_layout_is_marked_as_theirs(synth_mv):
    res = scan(_png(synth_mv, "6x2"), hints=ScanHints(layout="6x2"))
    assert res.layout.name == "6x2" and res.layout.confirmed_by_user


def test_the_device_registry_starts_empty_with_an_example():
    from ecgscan.config import config_dir, load_yaml

    assert device_profiles() == {}                                     # nothing invented
    example = _profile("_example", load_yaml("devices/_example.yaml"))
    assert (config_dir() / "devices" / "_example.yaml").exists()
    assert example.layouts == ("6x2",) and example.header == "bottom" and example.speeds_mm_s == (25.0, 50.0)


def test_a_device_profile_confirms_the_layout_the_page_shows(synth_mv, monkeypatch):
    from ecgscan import pipeline

    prof = DeviceProfile(id="machine-1", layouts=("6x2",))
    monkeypatch.setattr(pipeline, "device_profile", lambda pid: prof if pid == "machine-1" else None)
    dbg = ScanDebug()
    res = scan(_png(synth_mv, "6x2"), hints=ScanHints(device_profile="machine-1"), debug=dbg)
    assert res.layout.name == "6x2" and res.layout.device_profile_id == "machine-1" and res.layout.confidence >= 0.9
    # A profile never overrides what the page shows: a 3x4+rhythm page stays 3x4+rhythm.
    other = scan(_png(synth_mv, "3x4+rhythm"), hints=ScanHints(device_profile="machine-1"))
    assert other.layout.name == "3x4+rhythm" and other.layout.device_profile_id is None

