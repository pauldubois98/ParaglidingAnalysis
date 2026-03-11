#!/usr/bin/env python3
"""
Local development server for the Paragliding Heatmap project.

Replaces `python -m http.server 8000`.  Adds a /proxy endpoint so the
browser can reach external APIs (open-meteo ERA5, ESA WorldCover COG)
without hitting CORS restrictions.

Usage:
    python server.py          # serves on port 8000
    python server.py 9000     # custom port
"""

import sys
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
import requests

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

# Allowlist: only proxy these hosts (security guard for local dev)
ALLOWED_HOSTS = {
    'archive-api.open-meteo.com',
    'historical-forecast-api.open-meteo.com',
    'api.open-meteo.com',
    'esa-worldcover.s3.eu-central-1.amazonaws.com',
}


class Handler(SimpleHTTPRequestHandler):
    """Serves static files + a /proxy?url=<encoded> passthrough."""

    # ── CORS + no-cache headers on every response ─────────────────────
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Range, Content-Type')
        self.send_header('Access-Control-Expose-Headers', 'Content-Range, Accept-Ranges, Content-Length')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        if self.path.startswith('/proxy?') or self.path == '/proxy':
            self._handle_proxy()
        else:
            super().do_GET()

    # ── Proxy handler ─────────────────────────────────────────────────
    def _handle_proxy(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        url = params.get('url', [None])[0]

        if not url:
            self._error(400, 'Missing ?url= parameter')
            return

        host = urllib.parse.urlparse(url).hostname
        if host not in ALLOWED_HOSTS:
            self._error(403, f'Host not in allowlist: {host}')
            return

        # Forward Range header so geotiff.js can do COG range requests
        fwd_headers = {}
        if 'Range' in self.headers:
            fwd_headers['Range'] = self.headers['Range']

        try:
            resp = requests.get(url, headers=fwd_headers, timeout=30, stream=True)
            self.send_response(resp.status_code)
            for key in ('Content-Type', 'Content-Length',
                        'Content-Range', 'Accept-Ranges'):
                val = resp.headers.get(key)
                if val:
                    self.send_header(key, val)
            self.end_headers()
            self.wfile.write(resp.content)

        except Exception as e:
            print(f'[proxy] ERROR fetching {url}: {e}', file=sys.stderr)
            self._error(502, str(e))

    def _error(self, code, msg):
        body = msg.encode()
        self.send_response(code)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Suppress noisy tile fetches; only show proxy + errors
        if '/proxy' in args[0] or (len(args) > 1 and not args[1].startswith('2')):
            super().log_message(fmt, *args)


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


if __name__ == '__main__':
    server = ThreadingHTTPServer(('', PORT), Handler)
    print(f'Serving on http://localhost:{PORT}')
    print(f'Proxy allowlist: {", ".join(sorted(ALLOWED_HOSTS))}')
    print('Press Ctrl+C to stop.')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')
