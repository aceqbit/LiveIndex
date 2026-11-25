"""LiveIndex over its HTTP API only (no imports from the engine)."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.request

from liveindex_check.adapter import Ack, Hit, Result


class LiveIndexAdapter:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 30.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.secret = os.environ.get("LIVEINDEX_TOKEN_SECRET", "")

    def _call(self, method: str, path: str, body: dict | None = None, principal: str | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if principal is not None:
            if self.secret:
                sig = hmac.new(self.secret.encode(), principal.encode(), hashlib.sha256).hexdigest()
                req.add_header("Authorization", f"Bearer {principal}.{sig}")
            else:
                req.add_header("X-Test-Principal", principal)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read() or b"{}")

    def reset(self) -> None:
        self._call("POST", "/v1/admin/reset")

    def upsert(self, doc_id: str, text: str, allowed: set[str], tenant: str) -> Ack:
        body = {"doc_id": doc_id, "text": text, "tenant": tenant, "allowed": sorted(allowed)}
        e = self._call("POST", "/v1/docs", body)["epoch"]
        return Ack(str(e), e)

    def delete(self, doc_id: str) -> Ack:
        e = self._call("DELETE", f"/v1/docs/{doc_id}")["epoch"]
        return Ack(str(e), e)

    def set_acl(self, doc_id: str, allowed: set[str]) -> Ack:
        e = self._call("PUT", f"/v1/docs/{doc_id}/acl", {"allowed": sorted(allowed)})["epoch"]
        return Ack(str(e), e)

    def search(self, principal: str, query: str, k: int, min_ack: Ack | None) -> Result:
        body: dict = {"query": query, "k": k}
        if min_ack is not None and min_ack.epoch is not None:
            body["min_epoch"] = min_ack.epoch
        r = self._call("POST", "/v1/search", body, principal=principal)
        return Result([Hit(h["doc_id"], h["version"], h["chunk_text"]) for h in r["hits"]], r["epoch"])

    def settle(self, ack: Ack, timeout_s: float) -> bool:
        if ack.epoch is None:
            return True
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._call("GET", "/v1/epoch")["index_epoch"] >= ack.epoch:
                return True
            time.sleep(0.01)
        return False


def make(args) -> LiveIndexAdapter:
    return LiveIndexAdapter(getattr(args, "url", None) or "http://127.0.0.1:8000")
