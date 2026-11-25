"""The only interface the suite uses to talk to a system under test."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Protocol


@dataclass(frozen=True)
class Ack:
    """Opaque acknowledgement token; `epoch` is set when the system has epochs."""

    token: str
    epoch: int | None = None


class Hit(NamedTuple):
    doc_id: str
    version: int | None
    text: str


@dataclass
class Result:
    hits: list[Hit] = field(default_factory=list)
    epoch: int | None = None


class Adapter(Protocol):
    def reset(self) -> None: ...

    def upsert(self, doc_id: str, text: str, allowed: set[str], tenant: str) -> Ack: ...

    def delete(self, doc_id: str) -> Ack: ...

    def set_acl(self, doc_id: str, allowed: set[str]) -> Ack: ...

    def search(self, principal: str, query: str, k: int, min_ack: Ack | None) -> Result: ...

    def settle(self, ack: Ack, timeout_s: float) -> bool:
        """Wait until the write is visible, if the system has a notion of that."""
        ...
