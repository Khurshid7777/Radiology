"""D275: a picture whose enlarged copy is refused is read once more as it is."""
import pytest

import cardiolens.telegram.analyze as A


class _Prep:
    def __init__(self, path, enlarged):
        self.path, self.enlarged = path, enlarged

    def __call__(self, src, prep=None):
        return self

    def __enter__(self):
        return "ENLARGED.png" if self.enlarged else self.path

    def __exit__(self, *a):
        return False


class _Reduced:
    def __init__(self, small):
        self.small = small

    def __call__(self, src):
        return self

    def __enter__(self):
        return self.small

    def __exit__(self, *a):
        return False


def _patch(monkeypatch, enlarged, reader, small=None):
    import cardiolens.layer1_ingestion.image_prep as IP

    monkeypatch.setattr(IP, "prepared", _Prep("orig.png", enlarged))
    monkeypatch.setattr(IP, "reduced", _Reduced(small))
    monkeypatch.setattr(A, "_digitize_photo", reader)


def test_refused_enlarged_copy_falls_back_to_the_original(monkeypatch):
    calls = []

    def reader(path, **kw):
        calls.append(path)
        if path == "ENLARGED.png":
            raise ValueError("grid not found")
        return "READ-ORIGINAL"

    _patch(monkeypatch, True, reader)
    assert A.digitize_photo("orig.png") == "READ-ORIGINAL" and calls == ["ENLARGED.png", "orig.png"]


def test_original_refusal_is_the_one_shown(monkeypatch):
    def reader(path, **kw):
        raise ValueError("enlarged refused" if path == "ENLARGED.png" else "original refused")

    _patch(monkeypatch, True, reader)
    with pytest.raises(ValueError, match="original refused"):
        A.digitize_photo("orig.png")


def test_not_enlarged_picture_is_read_once(monkeypatch):
    calls = []

    def reader(path, **kw):
        calls.append(path)
        raise ValueError("refused")

    _patch(monkeypatch, False, reader)
    with pytest.raises(ValueError):
        A.digitize_photo("orig.png")
    assert calls == ["orig.png"]


def test_large_refused_picture_is_read_from_a_reduced_copy(monkeypatch):
    """A small photo stretched to 3000 x 4000: refused as it is, read from the reduced copy."""
    calls = []

    def reader(path, **kw):
        calls.append(path)
        if path == "reduced.png":
            return "READ-REDUCED"
        raise ValueError("grid not found")

    _patch(monkeypatch, False, reader, small="reduced.png")
    assert A.digitize_photo("orig.png") == "READ-REDUCED" and calls == ["orig.png", "reduced.png"]


def test_all_refused_shows_the_last_refusal(monkeypatch):
    def reader(path, **kw):
        raise ValueError({"ENLARGED.png": "enlarged", "orig.png": "original", "reduced.png": "reduced"}[path] + " refused")

    _patch(monkeypatch, True, reader, small="reduced.png")
    with pytest.raises(ValueError, match="reduced refused"):
        A.digitize_photo("orig.png")


def test_scanner_own_refusal_is_final_no_reduced_copy(monkeypatch):
    """A camera-sized photo with a coarse grid keeps the scanner's refusal (D170): no reduced copy is tried."""
    calls = []

    def reader(path, **kw):
        calls.append(path)
        raise A.ScannerRefusal("низкое разрешение")

    _patch(monkeypatch, False, reader, small="reduced.png")
    with pytest.raises(A.ScannerRefusal):
        A.digitize_photo("orig.png")
    assert calls == ["orig.png"]


def test_enlarged_scanner_refusal_still_reads_the_original_once(monkeypatch):
    calls = []

    def reader(path, **kw):
        calls.append(path)
        if path == "ENLARGED.png":
            raise A.ScannerRefusal("blur after the enlargement")
        return "READ-ORIGINAL"

    _patch(monkeypatch, True, reader, small="reduced.png")
    assert A.digitize_photo("orig.png") == "READ-ORIGINAL" and calls == ["ENLARGED.png", "orig.png"]
