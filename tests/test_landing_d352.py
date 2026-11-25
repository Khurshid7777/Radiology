"""D352: the landing page vision-ai.uz is served by the same Caddy, from a read-only host folder."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CADDYFILE = ROOT / "deploy/Caddyfile"


def site_blocks(text):
    """Top-level blocks of a Caddyfile: {address: body}; comments dropped."""
    text = "\n".join(line.split("#")[0].rstrip() for line in text.splitlines())
    blocks, depth, name, body = {}, 0, None, []
    for line in text.splitlines():
        if depth == 0:
            m = re.match(r"^(\S+)\s*\{\s*$", line)
            if m:
                name, body, depth = m.group(1), [], 1
            else:
                assert not line.strip(), f"unexpected top-level line: {line!r}"
            continue
        depth += line.count("{") - line.count("}")
        if depth == 0:
            assert line.strip() == "}"
            blocks[name] = [b.strip() for b in body if b.strip()]
        else:
            body.append(line)
    assert depth == 0, "unbalanced braces"
    return blocks


def test_caddyfile_serves_the_landing_and_leaves_the_app_block_alone():
    blocks = site_blocks(CADDYFILE.read_text())
    assert set(blocks) == {"{$DOMAIN}", "vision-ai.uz", "www.vision-ai.uz"}
    assert blocks["{$DOMAIN}"] == ["encode gzip", "request_body {", "max_size 110MB", "}", "reverse_proxy cardiolens:8000"]
    landing = blocks["vision-ai.uz"]
    assert sorted(landing) == ["encode gzip", "file_server", "root * /srv/landing"]     # static files only, no proxy
    assert blocks["www.vision-ai.uz"] == ["redir https://vision-ai.uz{uri} permanent"]


def test_landing_folder_is_mounted_read_only_from_landing_dir():
    comp = yaml.safe_load((ROOT / "deploy/docker-compose.yml").read_text())
    vols = comp["services"]["caddy"]["volumes"]
    assert "${LANDING_DIR:-/opt/landing}:/srv/landing:ro" in vols
    assert "./Caddyfile:/etc/caddy/Caddyfile:ro" in vols
    for name, svc in comp["services"].items():                  # only Caddy sees the landing folder
        if name != "caddy":
            assert not any("landing" in str(v).lower() for v in svc.get("volumes", [])), name
    env = (ROOT / "deploy/target.env").read_text()
    assert "\nLANDING_DIR=/opt/landing\n" in env and "\nDOMAIN=cardio.vision-ai.uz\n" in env


def run_server_env(tmp_path, target_env):
    """server_env.sh in a copy of deploy/ with an existing .env (so no keys are made); returns LANDING_DIR lines."""
    d = tmp_path / "deploy"
    d.mkdir()
    shutil.copy(ROOT / "deploy/server_env.sh", d / "server_env.sh")
    (d / "target.env").write_text(target_env)
    (d / ".env").write_text("DOMAIN=old.example\nECGSCAN_API_KEYS=clinic1:x,agents:y\nCARDIOLENS_SHARE_SECRET=z\n")
    r = subprocess.run(["bash", str(d / "server_env.sh"), "cardio.vision-ai.uz"], stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    return [line for line in (d / ".env").read_text().splitlines() if line.startswith("LANDING_DIR=")], r.stderr


@pytest.mark.parametrize("target, want, warned", [
    ("DOMAIN=cardio.vision-ai.uz\nLANDING_DIR=/opt/landing\n", "/opt/landing", False),
    ("LANDING_DIR=/srv/www/vision-ai.uz\r\n", "/srv/www/vision-ai.uz", False),
    ("DOMAIN=cardio.vision-ai.uz\n", "/opt/landing", False),                    # not set: the default
    ("LANDING_DIR=relative/dir\n", "/opt/landing", True),
    ("LANDING_DIR=/opt/../etc\n", "/opt/landing", True),
    ("LANDING_DIR=/opt/x|e touch /tmp/p\n", "/opt/landing", True),              # nothing odd reaches sed
    ("LANDING_DIR=/\n", "/opt/landing", True),
])
def test_server_env_passes_landing_dir_to_compose(tmp_path, target, want, warned):
    lines, err = run_server_env(tmp_path, target)
    assert lines == [f"LANDING_DIR={want}"]
    assert ("LANDING_DIR ignored" in err) is warned


def test_landing_files_are_not_in_git():
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.split("\n")
    assert not [f for f in tracked if f.split("/")[0] == "landing" or "/landing/" in f]


def test_caddy_validate_when_caddy_is_here():
    """The real check: `caddy validate` with the pinned image (CI has Docker) or a local caddy; skipped when neither."""
    if shutil.which("caddy"):
        cmd = ["caddy", "validate", "--config", str(CADDYFILE), "--adapter", "caddyfile"]
        env = {"DOMAIN": "cardio.vision-ai.uz", "PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp"}
    elif shutil.which("docker"):
        image = yaml.safe_load((ROOT / "deploy/docker-compose.yml").read_text())["services"]["caddy"]["image"]
        try:
            pulled = subprocess.run(["docker", "pull", "-q", image], capture_output=True, timeout=180, check=False)
        except subprocess.TimeoutExpired:
            pytest.skip("docker pull timed out")
        if pulled.returncode != 0:
            pytest.skip("the caddy image cannot be pulled here")
        cmd = ["docker", "run", "--rm", "-e", "DOMAIN=cardio.vision-ai.uz", "-v", f"{CADDYFILE}:/etc/caddy/Caddyfile:ro",
               image, "caddy", "validate", "--config", "/etc/caddy/Caddyfile", "--adapter", "caddyfile"]
        env = None
    else:
        pytest.skip("no caddy and no docker here")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
