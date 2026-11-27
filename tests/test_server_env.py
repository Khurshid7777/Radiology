"""D197: deploy/server_env.sh makes the share secret once, keeps it, and never prints it."""

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "deploy/server_env.sh"


def _run(tmp: Path) -> str:
    return subprocess.run(["bash", str(tmp / "server_env.sh"), "ecg.example.uz"], input="", text=True,
                          capture_output=True, check=True).stdout


def _secret(tmp: Path) -> str:
    line = next(x for x in (tmp / ".env").read_text().splitlines() if x.startswith("CARDIOLENS_SHARE_SECRET="))
    return line.split("=", 1)[1]


@pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl")
def test_the_secret_is_made_once_kept_and_never_printed(tmp_path):
    shutil.copy(SCRIPT, tmp_path / "server_env.sh")
    out1 = _run(tmp_path)
    s1 = _secret(tmp_path)
    assert len(s1) == 64 and all(c in "0123456789abcdef" for c in s1)
    assert s1 not in out1
    out2 = _run(tmp_path)
    assert _secret(tmp_path) == s1 and s1 not in out2 and "CARDIOLENS_SHARE_SECRET made" not in out2


@pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl")
def test_an_existing_env_without_the_secret_gets_one(tmp_path):
    shutil.copy(SCRIPT, tmp_path / "server_env.sh")
    (tmp_path / ".env").write_text("DOMAIN=x\nECGSCAN_API_KEYS=clinic1:abc\nCARDIOLENS_SHARE_SECRET=\n")
    _run(tmp_path)
    assert len(_secret(tmp_path)) == 64
    assert (tmp_path / ".env").read_text().count("CARDIOLENS_SHARE_SECRET=") == 1


def test_images_and_ci_install_pinned_versions():
    """D303: the site's image must not pick up a new release on its own (OpenCV 5.1 broke photo reading, D302)."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    pins = dict(ln.split("==") for ln in (root / "deploy/constraints.txt").read_text().splitlines() if "==" in ln)
    assert pins["opencv-python-headless"] == "5.0.0.93" and pins["opencv-python"] == "5.0.0.93"
    for must in ("numpy", "scipy", "neurokit2", "fastapi", "pydantic", "onnxruntime", "pillow", "aiogram"):
        assert must in pins, must
    for name in ("deploy/Dockerfile", "deploy/Dockerfile.agents"):
        text = (root / name).read_text()
        assert "COPY deploy/constraints.txt" in text
        installs = re.findall(r"pip install [^&\n]*-e ", text)
        assert installs and all("-c deploy/constraints.txt" in i for i in installs), name
    assert "-c deploy/constraints.txt" in (root / ".github/workflows/ci.yml").read_text()
