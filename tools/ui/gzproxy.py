"""Local stand-in for Caddy's `encode gzip`: serves PAGE at / gzipped, forwards the rest to the app."""
import gzip
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE, PORT, APP = sys.argv[1], int(sys.argv[2]), "http://127.0.0.1:8765"
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, body, ctype):
        z = gzip.compress(body); self.send_response(200); self.send_header("Content-Type", ctype)
        self.send_header("Content-Encoding", "gzip"); self.send_header("Content-Length", str(len(z))); self.end_headers(); self.wfile.write(z)
    def do_GET(self):
        if self.path in ("/", "/index.html"): return self._send(open(PAGE, "rb").read(), "text/html; charset=utf-8")
        try:
            with urllib.request.urlopen(APP + self.path, timeout=30) as r: self._send(r.read(), r.headers.get("Content-Type", "application/octet-stream"))
        except Exception: self.send_response(404); self.end_headers()
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
