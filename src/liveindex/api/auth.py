"""Principal from a verified bearer token (`<principal>.<hmac-sha256-hex>`), or a test header."""
from __future__ import annotations

import hashlib
import hmac
import os

from fastapi import HTTPException


def _secret() -> bytes:
    s = os.environ.get("LIVEINDEX_TOKEN_SECRET", "")
    if not s:
        raise HTTPException(401, "token verification not configured")
    return s.encode()


def sign(principal: str, secret: bytes | None = None) -> str:
    key = secret if secret is not None else _secret()
    return f"{principal}.{hmac.new(key, principal.encode(), hashlib.sha256).hexdigest()}"


def verify(token: str) -> str:
    principal, _, sig = token.rpartition(".")
    if not principal or not hmac.compare_digest(sign(principal), f"{principal}.{sig}"):
        raise HTTPException(401, "bad token")
    return principal


def resolve_principal(authorization: str | None, x_test_principal: str | None) -> str:
    """Never read the principal from the request body. The header is test-only."""
    if authorization and authorization.lower().startswith("bearer "):
        return verify(authorization[7:].strip())
    if os.environ.get("ALLOW_TEST_PRINCIPAL") == "1" and x_test_principal:
        return x_test_principal
    raise HTTPException(401, "no verified principal")
