"""Outbound text API boundary: explicit remote opt-in, no proxies or redirects."""
from __future__ import annotations

import os
import urllib.error
import urllib.request
from urllib.parse import urlsplit


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.URLError("API redirects are disabled to protect transcript privacy")


def api_urlopen(request: urllib.request.Request, timeout: float):
    parsed = urlsplit(request.full_url)
    if (parsed.scheme not in ("http", "https") or parsed.username is not None
            or parsed.password is not None or not parsed.hostname):
        raise urllib.error.URLError("Invalid text API URL")
    if (parsed.hostname not in ("localhost", "127.0.0.1", "::1", "0.0.0.0")
            and os.environ.get("DICTATE_ALLOW_REMOTE_API") != "1"):
        raise urllib.error.URLError("Remote text API requires DICTATE_ALLOW_REMOTE_API=1")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    return opener.open(request, timeout=timeout)
