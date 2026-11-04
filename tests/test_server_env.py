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
