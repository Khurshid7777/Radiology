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


LANDING = "vision-ai.uz, www.vision-ai.uz"


def test_landing_block_matches_the_owner_setup_and_app_site_unchanged():
    text = (DEPLOY / "Caddyfile").read_text()
    blocks = _blocks(text)
    assert set(blocks) == {"{$DOMAIN}", LANDING}
    app, landing = blocks["{$DOMAIN}"], blocks[LANDING]
    assert "reverse_proxy cardiolens:8000" in app and "max_size 110MB" in app
    # the owner's /opt/vision-ai/ensure-caddy.sh appends a second block unless a line starts with the domain;
    # two blocks for one domain stop Caddy (both sites down)
    assert re.search(r"(?m)^vision-ai\.uz", text)
    assert "root * /srv/vision-ai" in landing and "hide .*" in landing
    assert "cardiolens" not in landing
    assert re.findall(r"reverse_proxy\s+(\S+)", landing) == ["vision-ai-lead-bot:8081"]


def test_landing_files_mounted_read_only_from_the_owner_folder():
    compose = (DEPLOY / "docker-compose.yml").read_text()
    assert "- /var/www/vision-ai.uz:/srv/vision-ai:ro" in compose
    assert not (DEPLOY / "landing").exists()     # one copy of the site: the owner's folder on the server
