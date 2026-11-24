"""SD-137: capture on the Telegram Mini App page: the original photo goes to the server with a one-time
ticket, waits in memory and is taken once by the bot."""

import asyncio
import hashlib
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pytest

from ecgscan.i18n import strings
from ecgscan_service.miniapp import (
    TICKET_HEADER,
    Inbox,
    InboxFull,
    Ticket,
    image_kind,
    issue_ticket,
    turn_photo,
    verify_ticket,
)

cv2 = pytest.importorskip("cv2")
ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "src/ecgscan_service/static/miniapp_capture.html"
FIX = Path(__file__).parent / "fixtures" / "capture_qc"
KEYS = {"k-bot": "bot", "k-other": "other"}


# --- tickets ----------------------------------------------------------------------------------------

def test_a_ticket_is_accepted_only_live_untouched_and_signed_with_a_known_key():
    now = 1_800_000_000.0
    ticket, tid, expiry = issue_ticket("k-bot", 5, now)
    assert expiry == int(now + 300) and re.fullmatch(r"[0-9a-f]{24}", tid)
    assert verify_ticket(ticket, KEYS, max_minutes=10, now=now) == Ticket(tid, "bot", expiry)
    assert verify_ticket(ticket, {"k-other": "other"}, max_minutes=10, now=now) is None        # another signer
    assert verify_ticket(ticket, KEYS, max_minutes=10, now=expiry) is None                     # run out
    assert verify_ticket(ticket, KEYS, max_minutes=4, now=now) is None                         # too long a life
    head, exp, sig = ticket.split(".")
    for bad in (f"{head}.{int(exp) + 1}.{sig}", f"{'0' * 24}.{exp}.{sig}", f"{head}.{exp}.{'0' * 64}",
                ticket + "0", ticket.upper(), "", None, "a.b.c", f"{head}.{exp}"):
        assert verify_ticket(bad, KEYS, max_minutes=10, now=now) is None, bad
    assert issue_ticket("k-bot", 5, now)[1] != tid                                            # ids are random
    assert "k-bot" not in ticket


def test_the_inbox_gives_a_sent_file_once_to_its_owner_and_forgets_it_in_time():
    box, now = Inbox(max_waiting=2), 1000.0
    a, b, c = Ticket("a" * 24, "bot", 2000), Ticket("b" * 24, "bot", 1500), Ticket("c" * 24, "bot", 2000)
    assert box.send(a, now) is False                       # nothing uploaded yet
    box.put(a, b"first", now)
    box.put(a, b"second", now)                             # a retake replaces the earlier photo
    assert box.take(a.id, "bot", now) is None              # not sent yet
    assert box.send(a, now) is True
    assert box.take(a.id, "other", now) is None            # another key never gets it
    assert box.take(a.id, "bot", now) == [b"second"]
    assert box.take(a.id, "bot", now) is False and box.used(a.id, now)
    with pytest.raises(KeyError):
        box.put(a, b"again", now)                          # a used ticket takes no more files
    box.put(b, b"x", now)
    box.put(c, b"y", now)
    with pytest.raises(InboxFull):
        box.put(Ticket("d" * 24, "bot", 2000), b"z", now)
    assert len(box) == 2
    assert box.send(b, 1500.0) is False and len(box) == 1  # b ran out: the photo is gone
    assert box.used(a.id, 2000.0) is False                 # ... and so is the memory of a used ticket


# --- the service ------------------------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from ecgscan_service import router
    from ecgscan_service.app import app

    monkeypatch.setenv("ECGSCAN_API_KEYS", "bot:k-bot,other:k-other")
    monkeypatch.setenv("ECGSCAN_SERVICE_DIR", str(tmp_path / "service"))
    monkeypatch.chdir(tmp_path)
    router.get_inbox.cache_clear()
    yield TestClient(app)
    router.get_inbox.cache_clear()


def good_photo() -> bytes:
    """A full-size photo: the reference preview brought back to the size the phone took."""
    ref = json.loads((FIX / "clean_12.json").read_text())
    img = cv2.imread(str(FIX / "clean_12.jpg"))
    k = ref["scale_to_capture"]
    big = cv2.resize(img, (round(img.shape[1] * k), round(img.shape[0] * k)), interpolation=cv2.INTER_CUBIC)
    return cv2.imencode(".jpg", big, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


def ticket_headers(key: str = "k-bot", minutes: float = 5) -> tuple[dict, str]:
    ticket, tid, _ = issue_ticket(key, minutes)
    return {TICKET_HEADER: ticket}, tid


def test_the_original_reaches_the_bot_byte_for_byte_once_and_never_touches_the_disk(client, tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    photo = good_photo()
    head, tid = ticket_headers()
    bot = {"X-API-Key": "k-bot"}
    r = client.post("/miniapp/upload?lang=ru", content=photo, headers=head)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["checked"] and body["hints"] == [] and body["width"] > 3000 and body["px_per_mm"] > 8
    assert client.get(f"/scans/miniapp/inbox/{tid}", headers=bot).status_code == 204          # not sent yet
    assert client.post("/miniapp/send", headers=head).json() == {"sent": True}
    assert client.get(f"/scans/miniapp/inbox/{tid}", headers={"X-API-Key": "k-other"}).status_code == 204
    got = client.get(f"/scans/miniapp/inbox/{tid}", headers=bot)
    assert got.status_code == 200 and hashlib.sha256(got.content).digest() == hashlib.sha256(photo).digest()
    assert client.get(f"/scans/miniapp/inbox/{tid}", headers=bot).status_code == 404          # once
    assert client.post("/miniapp/upload", content=photo, headers=head).status_code == 401     # the ticket is spent
    assert client.post("/miniapp/send", headers=head).status_code == 401
    assert not any(p.is_file() for p in tmp_path.rglob("*"))                                  # memory only
    said = "\n".join(r.getMessage() for r in caplog.records if not r.name.startswith("httpx"))            # httpx: this test's own client
    assert tid not in said and head[TICKET_HEADER] not in said                                # ... and no trace in the logs


def test_the_bots_side_needs_the_clinic_key(client):
    _, tid = ticket_headers()
    assert client.get(f"/scans/miniapp/inbox/{tid}").status_code == 401
    assert client.get(f"/scans/miniapp/inbox/{tid}", headers={"X-API-Key": "nope"}).status_code == 401


@pytest.mark.parametrize("headers", [{}, {TICKET_HEADER: "x"}, {TICKET_HEADER: issue_ticket("k-stranger", 5)[0]},
                                     {TICKET_HEADER: issue_ticket("k-bot", 5, time.time() - 3600)[0]},
                                     {TICKET_HEADER: issue_ticket("k-bot", 600)[0]}, {"X-API-Key": "k-bot"}])
def test_without_a_good_ticket_nothing_is_accepted(client, headers):
    assert client.post("/miniapp/upload", content=b"\xff\xd8\xff", headers=headers).status_code == 401
    assert client.post("/miniapp/send", headers=headers).status_code == 401


def test_a_bad_ticket_is_refused_before_the_body_is_read(client):
    """An unauthenticated client must not be able to stream megabytes in before its 401 (as D136)."""
    from ecgscan_service.app import app

    sent = []

    async def receive():
        raise AssertionError("the body was read")

    async def send(msg):
        sent.append(msg)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "https",
             "path": "/miniapp/upload", "raw_path": b"/miniapp/upload", "query_string": b"", "root_path": "",
             "headers": [(b"x-miniapp-ticket", b"forged"), (b"content-type", b"application/octet-stream")],
             "client": ("203.0.113.1", 1), "server": ("site", 443)}
    asyncio.run(app(scope, receive, send))
    assert sent[0]["type"] == "http.response.start" and sent[0]["status"] == 401


def test_a_file_over_the_limit_is_cut_off(client, monkeypatch):
    from ecgscan_service import router

    real = router.load_yaml

    def small(name):
        cfg = real(name)
        return {**cfg, "upload": {**cfg["upload"], "max_mb": 0.001}} if name == "service" else cfg

    monkeypatch.setattr(router, "load_yaml", small)
    head, _ = ticket_headers()
    assert client.post("/miniapp/upload", content=b"x" * 2000, headers=head).status_code == 413
    assert client.post("/miniapp/send", headers=head).status_code == 409                       # nothing was kept
    assert client.post("/miniapp/upload", content=b"", headers=head).status_code == 400


def test_a_poor_photo_gets_the_frame_checks_words_in_the_pages_language(client):
    dark = (FIX / "dark.jpg").read_bytes()
    expected = json.loads((FIX / "dark.json").read_text())["expected_hints"]
    assert "light" in expected
    for lang in ("ru", "uz"):
        head, _ = ticket_headers()
        body = client.post(f"/miniapp/upload?lang={lang}", content=dark, headers=head).json()
        assert strings(lang)["qc.light"] in body["hints"], body
        # A poor photo is still the doctor's to send as it is.
        assert client.post("/miniapp/send", headers=head).status_code == 200
    # A small picture with no grid in it is called small; bytes that are no picture are let through unchecked.
    head, _ = ticket_headers()
    import numpy as np

    blank = cv2.imencode(".jpg", np.full((600, 800, 3), 128, np.uint8))[1].tobytes()
    body = client.post("/miniapp/upload?lang=ru", content=blank, headers=head).json()
    assert any("мелкий" in h for h in body["hints"]), body
    head, _ = ticket_headers()
    # bytes that start like a JPEG but have no readable header are refused before any decoder sees them
    assert client.post("/miniapp/upload?lang=xx", content=b"\xff\xd8\xff not a picture after all", headers=head).status_code == 415


def test_a_sheet_running_off_the_frame_is_told_which_way_to_move(client):
    """SD-133 on a taken photo: the left part of the sheet is cut off."""
    img = cv2.imdecode(__import__("numpy").frombuffer(good_photo(), "uint8"), cv2.IMREAD_COLOR)
    cut = img[:, img.shape[1] // 3:]
    head, _ = ticket_headers()
    body = client.post("/miniapp/upload?lang=ru", content=cv2.imencode(".jpg", cut)[1].tobytes(), headers=head).json()
    assert strings("ru")["qc.frame_left"] in body["hints"], body
    assert strings("ru")["qc.frame"] not in body["hints"]


# --- the page ---------------------------------------------------------------------------------------

FORBIDDEN = ("fetch(", "FormData", "sendBeacon", "WebSocket", "EventSource", "<form", "<iframe", "<link", "import(",
             "localStorage", "sessionStorage", "document.cookie", "tgWebAppData", "initData", "innerHTML", "eval(")


def test_the_page_talks_to_its_own_two_addresses_only():
    src = PAGE.read_text(encoding="utf-8")
    for word in FORBIDDEN:
        assert word not in src, word
    assert not re.search(r"https?://|//[a-z]", src.replace("// ", "").replace("/*", ""))        # no outside address
    assert "PAGE.telegram_web" in src and src.count("postMessage(") == 1                        # Telegram Web's frame, by the server's word
    assert "<script src" not in src
    assert set(re.findall(r"post\('([^']+)'", src)) == {"upload", "turn", "drop", "send"}     # SD-147: its own four addresses
    assert src.count("XMLHttpRequest") == 1 and src.count(".open(") == 1
    assert src.count('type="file"') == 2 and 'capture="environment"' in src
    assert TICKET_HEADER in src                                         # the page and the server name the header alike


@pytest.mark.parametrize("lang", ["ru", "uz"])
def test_the_page_is_served_open_with_its_texts_and_may_connect_to_itself_only(client, lang):
    r = client.get(f"/miniapp/capture?lang={lang}&t=abc")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    csp = r.headers["content-security-policy"]
    assert "default-src 'none'" in csp and "connect-src 'self'" in csp and "form-action 'none'" in csp
    assert csp.endswith("frame-ancestors https://web.telegram.org")     # only Telegram Web may frame the page
    assert "http" not in csp.replace("https://web.telegram.org", "") and "*" not in csp
    assert "img-src" not in csp and "media-src" not in csp              # nothing the page does not use
    head = client.head(f"/miniapp/capture?lang={lang}")
    assert head.status_code == 200 and head.headers["content-security-policy"] == csp
    assert r.headers["cache-control"] == "no-store" and r.headers["referrer-policy"] == "no-referrer"
    page = json.loads(re.search(r"const PAGE = (\{.*\});", r.text).group(1))
    s = strings(lang)
    assert page["lang"] == lang and page["max_mb"] == 25 and page["telegram_web"] == "https://web.telegram.org"
    assert page["texts"]["miniapp.shoot"] == s["miniapp.shoot"] and page["texts"]["miniapp.sent"] == s["miniapp.sent"]
    used = set(re.findall(r"text\('(miniapp\.[a-z_]+)'", PAGE.read_text(encoding="utf-8")))
    assert used and used <= set(page["texts"])                         # every text the page asks for is given
    assert "abc" not in r.text                                          # the ticket is read by the page, not echoed
    assert client.get("/miniapp/capture?lang=xx").status_code == 200    # an unknown language falls back to Russian


def test_page_texts_exist_in_both_languages():
    ru, uz = strings("ru"), strings("uz")
    keys = [k for k in ru if k.startswith("miniapp.")] + ["bot.scan", "bot.scan_btn", "bot.scan_off", "bot.scan_got"]
    assert len(keys) >= 20
    for k in keys:
        assert ru[k] and uz[k] and ru[k] != uz[k], k
        assert set(re.findall(r"\{(\w+)\}", ru[k])) == set(re.findall(r"\{(\w+)\}", uz[k])), k


def test_only_a_photo_the_scanner_reads_is_taken_whatever_the_client_calls_it(client):
    """The first bytes decide, not the name or the declared type; the rest of such a body is not read."""
    assert image_kind(b"\xff\xd8\xff\xe0") == "jpeg" and image_kind(b"\x89PNG\r\n\x1a\n") == "png"
    assert image_kind(b"II*\x00") == "tiff" and image_kind(b"MM\x00*") == "tiff"
    assert image_kind(b"%PDF-1.7") is None and image_kind(b"<html>") is None and image_kind(b"") is None
    head, _ = ticket_headers()
    for body in (b"%PDF-1.7 " + b"x" * 5000, b"<script>alert(1)</script>", b"MZ\x90\x00" + b"\x00" * 64, b"tiny",
                 b"RIFF\x00\x00\x00\x00WEBPVP8 "):
        r = client.post("/miniapp/upload", content=body, headers={**head, "Content-Type": "image/jpeg"})
        assert r.status_code == 415, body[:8]
    assert client.post("/miniapp/send", headers=head).status_code == 409                       # nothing was kept
    from ecgscan_service.app import app

    calls, sent = [], []

    async def receive():
        calls.append(1)
        assert len(calls) == 1, "the body was read on after its first bytes said it is no photo"
        return {"type": "http.request", "body": b"%PDF-1.7 not a photo", "more_body": True}

    async def send(msg):
        sent.append(msg)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "https",
             "path": "/miniapp/upload", "raw_path": b"/miniapp/upload", "query_string": b"", "root_path": "",
             "headers": [(b"x-miniapp-ticket", ticket_headers()[0][TICKET_HEADER].encode())],
             "client": ("203.0.113.1", 1), "server": ("site", 443)}
    asyncio.run(app(scope, receive, send))
    assert sent[0]["status"] == 415 and len(calls) == 1


def test_one_button_takes_a_limited_number_of_retakes(client, monkeypatch):
    from ecgscan_service import router

    real = router.load_yaml

    def few(name):
        cfg = real(name)
        return {**cfg, "miniapp": {**cfg["miniapp"], "max_uploads_per_ticket": 2}} if name == "service" else cfg

    monkeypatch.setattr(router, "load_yaml", few)
    import numpy as np

    photo = cv2.imencode(".jpg", np.full((400, 600, 3), 255, np.uint8))[1].tobytes()
    head, _ = ticket_headers()
    other, _ = ticket_headers()
    assert [client.post("/miniapp/upload", content=photo, headers=head).status_code for _ in range(3)] == [200, 200, 401]
    assert client.post("/miniapp/upload", content=photo, headers=other).status_code == 200    # counted per ticket
    assert client.post("/miniapp/send", headers=head).json() == {"sent": True}                 # the last good one goes

    box = Inbox(5)
    tk = Ticket("a" * 24, "bot", 2000)
    assert [box.attempt(tk, 2, 1000.0) for _ in range(3)] == [True, True, False]
    assert box.attempt(tk, 2, 2000.0) is True                                                  # forgotten with the ticket


def test_a_restart_loses_the_waiting_photo_but_not_the_button(client):
    """Tickets are checked by signature: after a restart the page uploads again with the same one."""
    import numpy as np

    from ecgscan_service import router

    photo = cv2.imencode(".jpg", np.full((400, 600, 3), 255, np.uint8))[1].tobytes()
    head, tid = ticket_headers()
    assert client.post("/miniapp/upload", content=photo, headers=head).status_code == 200
    router.get_inbox.cache_clear()                                                             # the process restarted
    assert client.post("/miniapp/send", headers=head).status_code == 409                       # the page offers to repeat
    assert client.get(f"/scans/miniapp/inbox/{tid}", headers={"X-API-Key": "k-bot"}).status_code == 204
    assert client.post("/miniapp/upload", content=photo, headers=head).status_code == 200
    assert client.post("/miniapp/send", headers=head).json() == {"sent": True}
    assert client.get(f"/scans/miniapp/inbox/{tid}", headers={"X-API-Key": "k-bot"}).content == photo


# --- SD-138: a sheet that stands on end ---------------------------------------------------------------

def marked(w: int = 600, h: int = 400) -> bytes:
    """A light picture with a dark square in its top left corner."""
    import numpy as np

    img = np.full((h, w, 3), 230, np.uint8)
    img[: h // 4, : w // 4] = 20
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()


def dark_corner(data: bytes) -> tuple[str, tuple[int, int]]:
    """Which corner of the picture is dark, and its (width, height)."""
    import numpy as np

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    q = {"top-left": img[: h // 8, : w // 8], "top-right": img[: h // 8, -w // 8:],
         "bottom-left": img[-h // 8:, : w // 8], "bottom-right": img[-h // 8:, -w // 8:]}
    return min(q, key=lambda k: q[k].mean()), (w, h)


def test_the_upload_says_when_to_ask_the_doctor_which_way_is_up(client):
    import numpy as np

    good = good_photo()
    assert client.post("/miniapp/upload", content=good, headers=ticket_headers()[0]).json()["turn"] is False
    img = cv2.imdecode(np.frombuffer(good, np.uint8), cv2.IMREAD_COLOR)
    for k in (1, 3):                                                  # the same sheet standing on end, either way
        side = cv2.imencode(".jpg", np.ascontiguousarray(np.rot90(img, k)), [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
        body = client.post("/miniapp/upload", content=side, headers=ticket_headers()[0]).json()
        assert body["turn"] is True and body["checked"], k
    # no sheet found: the photo's own shape decides (the core's portrait check, D171)
    assert client.post("/miniapp/upload", content=marked(400, 600), headers=ticket_headers()[0]).json()["turn"] is True
    assert client.post("/miniapp/upload", content=marked(600, 400), headers=ticket_headers()[0]).json()["turn"] is False
    assert client.post("/miniapp/upload", content=marked(600, 600), headers=ticket_headers()[0]).json()["turn"] is False


@pytest.mark.parametrize("rotate, corner, size", [(90, "top-right", (400, 600)), (180, "bottom-right", (600, 400)),
                                                  (270, "bottom-left", (400, 600))])
def test_the_doctors_turn_reaches_the_bot_as_a_turned_picture(client, rotate, corner, size):
    head, tid = ticket_headers()
    assert client.post("/miniapp/upload", content=marked(), headers=head).status_code == 200
    assert client.post(f"/miniapp/send?rotate={rotate}", headers=head).json() == {"sent": True}
    assert client.post(f"/miniapp/send?rotate={rotate}", headers=head).json() == {"sent": True}   # a repeat does not turn twice
    got = client.get(f"/scans/miniapp/inbox/{tid}", headers={"X-API-Key": "k-bot"})
    assert got.status_code == 200 and got.content[:3] == b"\xff\xd8\xff"
    assert dark_corner(got.content) == (corner, size)


def test_a_turn_that_is_not_a_quarter_is_refused_and_no_turn_leaves_the_original(client):
    head, tid = ticket_headers()
    photo = marked()
    assert client.post("/miniapp/upload", content=photo, headers=head).status_code == 200
    for bad in ("45", "-90", "360", "left"):
        assert client.post(f"/miniapp/send?rotate={bad}", headers=head).status_code == 422, bad
    assert client.get(f"/scans/miniapp/inbox/{tid}", headers={"X-API-Key": "k-bot"}).status_code == 204   # nothing was sent
    assert client.post("/miniapp/send?rotate=0", headers=head).json() == {"sent": True}
    assert client.get(f"/scans/miniapp/inbox/{tid}", headers={"X-API-Key": "k-bot"}).content == photo


def test_turning_a_photo():
    photo = marked()
    assert turn_photo(photo, 0, 95, 10**7) is photo
    assert dark_corner(turn_photo(photo, 90, 95, 10**7)) == ("top-right", (400, 600))
    for bad in (45, -90, 360):
        with pytest.raises(ValueError):
            turn_photo(photo, bad, 95, 10**7)
    with pytest.raises(ValueError):
        turn_photo(b"\xff\xd8\xff not a picture", 90, 95, 10**7)
    big = len(turn_photo(photo, 90, 95, 10**7))
    small = turn_photo(photo, 90, 95, big - 1)                        # does not fit: a lower quality is taken
    assert len(small) < big and dark_corner(small)[0] == "top-right"
    with pytest.raises(ValueError):
        turn_photo(photo, 90, 95, 100)


def test_a_photo_is_turned_from_the_way_the_doctor_saw_it():
    """A phone held upright writes the turn into EXIF; the page's small copies and the server both apply it."""
    import io

    import numpy as np
    pil = pytest.importorskip("PIL.Image")

    img = cv2.cvtColor(cv2.imdecode(np.frombuffer(marked(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    exif = pil.Exif()
    exif[0x0112] = 6                                                  # shown turned a quarter clockwise
    pil.fromarray(img).save(buf, format="JPEG", quality=95, exif=exif)
    assert dark_corner(buf.getvalue()) == ("top-right", (400, 600))   # as shown: the mark is top right
    assert dark_corner(turn_photo(buf.getvalue(), 90, 95, 10**7)) == ("bottom-right", (600, 400))


def test_the_turn_choice_is_on_the_page():
    src = PAGE.read_text(encoding="utf-8")
    for part in ('id="turnbox"', 'id="b_cw"', 'id="b_ccw"', 'id="b_keep"', "createImageBitmap(", "pick(90)", "pick(270)",
                 "pick(0)", "'&rotate=' + deg"):
        assert part in src, part
    assert "toBlob" not in src and "toDataURL" not in src            # the page shows the turn, the server makes it


# --- the core's review of SD-137/138: a declared size costs memory, a read-only key signs nothing -------

def png_declaring(w: int, h: int) -> bytes:
    """A tiny PNG whose header declares w x h px (the decoder would allocate that much)."""
    import struct
    import zlib

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\x00" * 16)) + chunk(b"IEND", b"")


def test_a_picture_declaring_a_huge_size_is_refused_before_it_is_decoded(client, monkeypatch):
    from ecgscan_service.miniapp import size_refusal

    bomb = png_declaring(30000, 30000)
    assert len(bomb) < 1000 and size_refusal(bomb) == "IMAGE_TOO_LARGE"
    assert size_refusal(png_declaring(50, 50)) == "LOW_RESOLUTION" and size_refusal(good_photo()) is None
    decoded = []
    real = cv2.imdecode
    monkeypatch.setattr(cv2, "imdecode", lambda *a, **k: decoded.append(1) or real(*a, **k))
    head, _ = ticket_headers()
    assert client.post("/miniapp/upload", content=bomb, headers=head).status_code == 413
    assert client.post("/miniapp/send", headers=head).status_code == 409                     # nothing was kept
    assert not decoded                                                                       # never decoded
    with pytest.raises(ValueError):
        turn_photo(bomb, 90, 95, 10**7)
    assert not decoded


def test_a_read_only_key_never_signs_a_ticket(monkeypatch, client):
    monkeypatch.setenv("ECGSCAN_API_KEYS", "bot:k-bot,agents:k-agents")
    head, _ = ticket_headers("k-agents")
    assert client.post("/miniapp/upload", content=b"\xff\xd8\xff", headers=head).status_code == 401
    assert client.post("/miniapp/send", headers=head).status_code == 401
    ticket, _, _ = issue_ticket("k-agents", 5)
    keys = {"k-bot": "bot", "k-agents": "agents"}
    assert verify_ticket(ticket, keys, max_minutes=10, refused_prefix=None) is not None      # signed by a known key ...
    assert verify_ticket(ticket, keys, max_minutes=10) is None                               # refused by default too
    assert verify_ticket(ticket, keys, max_minutes=10, refused_prefix="agent") is None       # ... but not by a read-only one
    assert verify_ticket(issue_ticket("k-bot", 5)[0], keys, max_minutes=10, refused_prefix="agent") is not None


def test_a_header_the_reader_does_not_know_is_refused_before_decoding(client, monkeypatch):
    """Some TIFF headers PIL cannot read (a palette without a ColorMap...) still expand to gigabytes in OpenCV."""
    from ecgscan_service.miniapp import size_refusal

    tiff = b"II*\x00\x08\x00\x00\x00" + b"\x00" * 32
    assert size_refusal(tiff) == "UNREADABLE_HEADER" and size_refusal(b"\xff\xd8\xff nothing") == "UNREADABLE_HEADER"
    decoded = []
    real = cv2.imdecode
    monkeypatch.setattr(cv2, "imdecode", lambda *a, **k: decoded.append(1) or real(*a, **k))
    head, _ = ticket_headers()
    assert client.post("/miniapp/upload", content=tiff, headers=head).status_code == 415
    assert client.post("/miniapp/send", headers=head).status_code == 409
    with pytest.raises(ValueError):
        turn_photo(tiff, 90, 95, 10**7)
    assert not decoded


def test_frame_checks_run_on_their_own_two_workers():
    from ecgscan_service import router

    pool = router._hint_pool()
    assert pool is router._hint_pool() and pool._max_workers == 2


# --- SD-147: several photos of one record --------------------------------------------------------------

def small_photo(shade: int) -> bytes:
    import numpy as np

    return cv2.imencode(".jpg", np.full((400, 600, 3), shade, np.uint8))[1].tobytes()


def take_parts(client, tid):
    r = client.get(f"/scans/miniapp/inbox/{tid}", headers={"X-API-Key": "k-bot"})
    assert r.status_code == 200
    sizes = [int(x) for x in r.headers["x-miniapp-sizes"].split(",")]
    out, at = [], 0
    for n in sizes:
        out.append(r.content[at:at + n])
        at += n
    assert at == len(r.content)
    return out


def test_nine_photos_of_a_strip_reach_the_bot_in_order_byte_for_byte(client):
    head, tid = ticket_headers()
    photos = [small_photo(100 + 10 * i) for i in range(9)]
    for i, ph in enumerate(photos):
        r = client.post("/miniapp/upload", content=ph, headers=head)
        assert r.status_code == 200 and r.json()["part"] == i
    assert client.post("/miniapp/send", headers=head).json() == {"sent": True}
    assert take_parts(client, tid) == photos
    assert client.get(f"/scans/miniapp/inbox/{tid}", headers={"X-API-Key": "k-bot"}).status_code == 404


def test_each_photo_has_its_own_turn_and_a_photo_can_be_dropped(client):
    head, tid = ticket_headers()
    a, b, c = marked(), marked(), small_photo(200)
    for ph in (a, b, c):
        assert client.post("/miniapp/upload", content=ph, headers=head).status_code == 200
    assert client.post("/miniapp/turn?part=1&rotate=90", headers=head).json() == {"part": 1, "rotate": 90}
    assert client.post("/miniapp/turn?part=1&rotate=45", headers=head).status_code == 422
    assert client.post("/miniapp/turn?part=7&rotate=90", headers=head).status_code == 404
    assert client.post("/miniapp/drop?part=2", headers=head).json() == {"dropped": 2}         # the last one goes
    assert client.post("/miniapp/drop?part=5", headers=head).status_code == 404
    assert client.post("/miniapp/send", headers=head).json() == {"sent": True}
    got = take_parts(client, tid)
    assert len(got) == 2 and got[0] == a                                                     # no turn: the original
    assert dark_corner(got[1]) == ("top-right", (400, 600))                                  # its own quarter turn


def test_a_record_has_limits_per_button_and_for_the_whole_store(client, monkeypatch):
    from ecgscan_service import router

    real = router.load_yaml

    def small(name):
        cfg = real(name)
        if name != "service":
            return cfg
        return {**cfg, "miniapp": {**cfg["miniapp"], "max_files": 3, "max_total_mb": 0.05, "max_store_mb": 0.06}}

    monkeypatch.setattr(router, "load_yaml", small)
    router.get_inbox.cache_clear()
    head, _ = ticket_headers()
    ph = small_photo(120)
    assert [client.post("/miniapp/upload", content=ph, headers=head).status_code for _ in range(4)] == [200, 200, 200, 413]
    big = cv2.imencode(".png", np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8))[1].tobytes()
    assert len(big) > 0.05 * 1024 * 1024
    assert client.post("/miniapp/upload", content=big, headers=ticket_headers()[0]).status_code == 413   # over a record's bytes
    router.get_inbox.cache_clear()
    from ecgscan_service.miniapp import TooMuch

    box = Inbox(5, max_files=10, max_ticket_bytes=100, max_store_bytes=150)                  # the whole store's limit
    t1, t2 = Ticket("a" * 24, "bot", 2000), Ticket("b" * 24, "bot", 2000)
    assert box.add(t1, b"x" * 90, 1000.0) == 0
    with pytest.raises(TooMuch):
        box.add(t1, b"x" * 20, 1000.0)                                                       # over the record's 100
    with pytest.raises(InboxFull):
        box.add(t2, b"x" * 70, 1000.0)                                                       # over the store's 150
    assert box.add(t2, b"x" * 50, 1000.0) == 0


def test_the_page_lets_several_photos_be_chosen_and_sends_them_together():
    src = PAGE.read_text(encoding="utf-8")
    assert 'id="f_gal" type="file" accept="image/*" multiple' in src
    assert "parts.length >= MAX" in src and "miniapp.send_n" in src and "dropLast" in src
    assert "await send()" not in src                                                         # no automatic send any more
    assert "p.codes[k] === 'frame'" in src                                                   # parts are not told the sheet is cut


# --- SD-147, the core's review: room is taken before the body is read ------------------------------------

def concurrent_uploads(app, tickets, declared):
    """Uploads at once; each body is held until all have started. Returns (statuses, the uploads whose body was read)."""
    read, sent = set(), {}

    async def one(i, ticket):
        gate = asyncio.Event()
        started.append(gate)

        async def receive():
            read.add(i)
            await gate.wait()
            return {"type": "http.request", "body": b"\xff\xd8\xff" + b"\x00" * 13, "more_body": False}

        async def send(msg):
            if msg["type"] == "http.response.start":
                sent[i] = msg["status"]

        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "https",
                 "path": "/miniapp/upload", "raw_path": b"/miniapp/upload", "query_string": b"", "root_path": "",
                 "headers": [(b"x-miniapp-ticket", ticket.encode()), (b"content-length", str(declared).encode())],
                 "client": ("203.0.113.1", 1), "server": ("site", 443)}
        await app(scope, receive, send)

    started = []

    async def go():
        tasks = [asyncio.create_task(one(i, t)) for i, t in enumerate(tickets)]
        await asyncio.sleep(0.3)
        for g in started:
            g.set()
        await asyncio.gather(*tasks)

    asyncio.run(go())
    return sent, read


def test_thirty_big_uploads_on_one_button_read_only_what_the_record_can_hold(client):
    from ecgscan_service.app import app

    ticket = ticket_headers()[0][TICKET_HEADER]
    sent, read = concurrent_uploads(app, [ticket] * 30, 25 * 1024 * 1024)
    assert len(read) <= 2                                       # 60 MB a record: two 25-MB bodies at most are read
    assert set(sent.values()) <= {200, 400, 401, 413, 415}      # 401: the button's upload count, also before reading
    assert sum(1 for v in sent.values() if v in (401, 413)) >= 28


def test_uploads_in_flight_on_the_whole_service_are_limited_before_reading(client):
    from ecgscan_service.app import app

    tickets = [ticket_headers()[0][TICKET_HEADER] for _ in range(6)]
    sent, read = concurrent_uploads(app, tickets, 1000)
    assert len(read) <= 4 and sum(1 for v in sent.values() if v == 429) >= 2       # max_inflight: 4


def test_the_turned_record_still_fits_its_limit(client, monkeypatch):
    from ecgscan_service import router

    real = router.load_yaml
    photos = [marked(1200, 800) for _ in range(3)]
    limit = sum(len(p) for p in photos) + 2000

    def tight(name):
        cfg = real(name)
        return {**cfg, "miniapp": {**cfg["miniapp"], "max_total_mb": limit / 1024 / 1024}} if name == "service" else cfg

    monkeypatch.setattr(router, "load_yaml", tight)
    router.get_inbox.cache_clear()
    head, tid = ticket_headers()
    for i, ph in enumerate(photos):
        assert client.post("/miniapp/upload", content=ph, headers={**head, "Content-Length": str(len(ph))}).status_code == 200
        assert client.post(f"/miniapp/turn?part={i}&rotate=90", headers=head).status_code == 200
    assert client.post("/miniapp/send", headers=head).json() == {"sent": True}
    got = take_parts(client, tid)
    assert len(got) == 3 and sum(len(g) for g in got) <= limit
    router.get_inbox.cache_clear()


def test_a_full_record_is_told_apart_from_a_big_file(client, monkeypatch):
    from ecgscan_service import router

    real = router.load_yaml

    def two(name):
        cfg = real(name)
        return {**cfg, "miniapp": {**cfg["miniapp"], "max_files": 1}} if name == "service" else cfg

    monkeypatch.setattr(router, "load_yaml", two)
    router.get_inbox.cache_clear()
    head, _ = ticket_headers()
    assert client.post("/miniapp/upload", content=small_photo(90), headers=head).status_code == 200
    r = client.post("/miniapp/upload", content=small_photo(91), headers=head)
    assert r.status_code == 413 and "record" in r.json()["detail"]
    src = PAGE.read_text(encoding="utf-8")
    assert "miniapp.record_full" in src and "/record/.test(" in src
    router.get_inbox.cache_clear()
