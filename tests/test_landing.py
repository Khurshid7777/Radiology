"""D352: vision-ai.uz serves the owner's static landing; the app site is unchanged."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"


def _blocks(caddyfile: str) -> dict[str, str]:
    """Top-level site blocks of the Caddyfile: address -> body."""
    out, depth, addr, body = {}, 0, None, []
    for line in caddyfile.splitlines():
        s = line.split("#")[0].rstrip()
        if not s.strip():
            continue
        if depth == 0 and s.endswith("{"):
            addr, body = s[:-1].strip(), []
        elif depth > 0:
            body.append(s)
        depth += s.count("{") - s.count("}")
        if depth == 0 and addr is not None:
            out[addr] = "\n".join(body)
            addr = None
    return out


def test_landing_is_static_and_app_site_unchanged():
    blocks = _blocks((DEPLOY / "Caddyfile").read_text())
    assert set(blocks) == {"{$DOMAIN}", "{$LANDING_DOMAIN:vision-ai.uz}"}
    app, landing = blocks["{$DOMAIN}"], blocks["{$LANDING_DOMAIN:vision-ai.uz}"]
    assert "reverse_proxy cardiolens:8000" in app and "max_size 110MB" in app
    assert "file_server" in landing and "hide .*" in landing and "root * /srv/landing" in landing
    assert "reverse_proxy" not in landing and "cardiolens" not in landing


def test_landing_mounted_read_only():
    compose = (DEPLOY / "docker-compose.yml").read_text()
    assert "- ./landing:/srv/landing:ro" in compose


def test_landing_files_present_and_clean():
    land = DEPLOY / "landing"
    for page in ("index.html", "uz/index.html", "en/index.html", "ru/index.html", "sitemap.xml"):
        assert (land / page).is_file(), page
    allowed = {".html", ".xml", ".svg", ".png", ".webmanifest", ".txt", ".ico"}
    for f in land.rglob("*"):
        if f.is_file():
            assert f.suffix in allowed, f
            if f.suffix in {".html", ".xml"}:
                text = f.read_text()
                assert not re.search(r"(?i)(api[_-]?key|bot\d+:|sk-ant|password)", text), f
