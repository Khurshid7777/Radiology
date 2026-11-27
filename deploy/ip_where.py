#!/usr/bin/env python3
"""Where an IP address is (D310): country, city, provider / AS and the abuse e-mail, for the daily security check.

    python3 deploy/ip_where.py CACHE.json IP [IP ...]   -> one line per IP: "IP  country, city — provider; abuse: e-mail"

Standard library only (runs on the host). Sources: ipinfo.io (no key) and RDAP (rdap.org). Only the IP addresses
of login attempts leave the server. Results are cached for 7 days; at most MAX_LOOKUPS network lookups per run;
any network error gives "нет данных" and never fails the check.
"""
from __future__ import annotations

import ipaddress
import json
import re
import sys
import time
import urllib.request

TTL_S = 7 * 24 * 3600
MAX_LOOKUPS = 15
TIMEOUT_S = 5


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "server-security-check"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:      # noqa: S310 (fixed https hosts)
        return json.loads(r.read(200_000).decode("utf-8", "replace"))


def parse_ipinfo(d: dict) -> dict:
    org = d.get("org") or ""
    return {"country": d.get("country") or "", "region": d.get("region") or "", "city": d.get("city") or "",
            "org": org, "host": d.get("hostname") or ""}


def parse_rdap_abuse(d: dict) -> str:
    """The first e-mail of an entity with the role 'abuse' (searched in nested entities too)."""
    stack = list(d.get("entities") or [])
    while stack:
        e = stack.pop(0)
        if "abuse" in (e.get("roles") or []):
            for item in (e.get("vcardArray") or [None, []])[1]:
                if item and item[0] == "email" and len(item) >= 4:
                    return str(item[3])
        stack.extend(e.get("entities") or [])
    return ""


def lookup(ip: str, get=_get) -> dict:
    out = {"t": time.time()}
    try:
        out.update(parse_ipinfo(get(f"https://ipinfo.io/{ip}/json")))
    except Exception:  # noqa: BLE001 - any failure is "no data"
        out["err"] = True
    try:
        out["abuse"] = parse_rdap_abuse(get(f"https://rdap.org/ip/{ip}"))
    except Exception:  # noqa: BLE001
        out.setdefault("abuse", "")
    return out


_SAFE = re.compile(r"[^A-Za-z0-9 .,@_()+:-]")


def _clean(text: str) -> str:
    """Text from outside (a PTR name the attacker sets, registry data) reaches an LLM agent's input: plain chars only."""
    return _SAFE.sub("", str(text))[:100]


def describe(ip: str, info: dict | None) -> str:
    if not info or (info.get("err") and not info.get("abuse")):
        return f"{ip}  нет данных о месте"
    place = ", ".join(_clean(x) for x in (info.get("country"), info.get("region"), info.get("city")) if x) or "место неизвестно"
    org = _clean(info.get("org") or "") or "провайдер неизвестен"
    extra = f"; abuse: {_clean(info['abuse'])}" if info.get("abuse") else ""
    host = f"; {_clean(info['host'])}" if info.get("host") else ""
    return f"{ip}  {place} — {org}{host}{extra}"


def where(ips: list[str], cache_path: str, get=_get, now: float | None = None) -> list[str]:
    now = time.time() if now is None else now
    try:
        with open(cache_path, encoding="utf-8") as fh:
            cache = json.load(fh)
    except (OSError, ValueError):
        cache = {}
    lookups, lines = 0, []
    for ip in ips:
        try:
            if not ipaddress.ip_address(ip).is_global:
                lines.append(f"{ip}  внутренний адрес")
                continue
        except ValueError:
            continue
        info = cache.get(ip)
        ttl = 86400 if info and info.get("err") else TTL_S          # a failed lookup is retried the next day
        if (not info or now - info.get("t", 0) > ttl) and lookups < MAX_LOOKUPS:
            info = lookup(ip, get)
            lookups += 1
            cache[ip] = info
        lines.append(describe(ip, info))
    try:
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump({k: v for k, v in cache.items() if now - v.get("t", 0) <= TTL_S}, fh)
    except OSError:
        pass
    return lines


if __name__ == "__main__":
    for line in where(sys.argv[2:], sys.argv[1]):
        print(line)
