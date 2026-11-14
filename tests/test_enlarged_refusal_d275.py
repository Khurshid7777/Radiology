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


def _patch(monkeypatch, enlarged, reader):
    import cardiolens.layer1_ingestion.image_prep as IP

    monkeypatch.setattr(IP, "prepared", _Prep("orig.png", enlarged))
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
