"""The text API cannot leak transcripts through remote URLs or redirects."""
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch
from urllib.error import URLError
from urllib.request import Request

import pytest

from dictate.api_http import api_urlopen


def test_remote_url_rejected_before_network(monkeypatch):
    monkeypatch.delenv('DICTATE_ALLOW_REMOTE_API', raising=False)
    with patch('dictate.api_http.urllib.request.build_opener') as opener:
        with pytest.raises(URLError, match='Remote text API'):
            api_urlopen(Request('https://example.com/v1/chat/completions', data=b'private'), 1)
        opener.assert_not_called()


def test_local_redirect_is_not_followed(monkeypatch):
    visited = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            visited.append(self.path)
            self.send_response(307)
            self.send_header('Location', '/leak')
            self.end_headers()
        def log_message(self, *args):
            pass
    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv('HTTP_PROXY', 'http://127.0.0.1:1')
    try:
        with pytest.raises(URLError, match='redirects are disabled'):
            api_urlopen(Request(f'http://127.0.0.1:{server.server_port}/start', data=b'private'), 2)
        assert visited == ['/start']
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
