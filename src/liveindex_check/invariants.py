"""Ground truth model and the invariant checks I1 to I5. Pure functions: no I/O."""
from __future__ import annotations

from dataclasses import dataclass, field

from liveindex_check.adapter import Ack, Hit, Result
from liveindex_check.docgen import parse_marker


@dataclass
class Violation:
    invariant: str  # I1..I5 or PRESENCE
    detail: str


@dataclass
class DocState:
    version: int = 0
    texts: dict[int, str] = field(default_factory=dict)
    live: bool = False
    allowed: set[str] = field(default_factory=set)
    ever_allowed: set[str] = field(default_factory=set)
    acl_ack: Ack | None = None  # ack of the last ACL change (or delete)


@dataclass
class Truth:
    docs: dict[str, DocState] = field(default_factory=dict)
    # things that must stay invisible: (kind, doc_id, version, principal | None, query_text)
    tombstones: list[tuple[str, str, int, str | None, str]] = field(default_factory=list)

    def live_docs(self) -> list[str]:
        return sorted(d for d, s in self.docs.items() if s.live)


def hit_version(h: Hit) -> int | None:
    if h.version is not None:
        return h.version
    pm = parse_marker(h.text)
    return pm[1] if pm and pm[0] == h.doc_id else None


def check_i1(doc_id: str, result: Result) -> list[Violation]:
    """After an acknowledged delete (settled), no chunk of the document is returned."""
    return [Violation("I1", f"deleted doc {h.doc_id} returned") for h in result.hits if h.doc_id == doc_id][:1]


def check_i2_old(doc_id: str, old_version: int, current: int, result: Result) -> list[Violation]:
    """After an acknowledged update, no old-version chunk is returned."""
    for h in result.hits:
        v = hit_version(h)
        if h.doc_id == doc_id and v is not None and v < current:
            return [Violation("I2", f"stale {doc_id} v{v} returned (current v{current})")]
    return []


def check_i2_new(doc_id: str, version: int, result: Result) -> list[Violation]:
    """The new version's text is retrievable."""
    if any(h.doc_id == doc_id and hit_version(h) == version for h in result.hits):
        return []
    return [Violation("I2", f"new {doc_id} v{version} not retrievable after update")]


def check_i3(doc_id: str, result: Result) -> list[Violation]:
    """After an acknowledged revoke (settled), the principal never gets a chunk of that document."""
    return [Violation("I3", f"revoked principal still gets {h.doc_id}") for h in result.hits if h.doc_id == doc_id][:1]


def check_i4(truth: Truth, principal: str, result: Result) -> list[Violation]:
    """Every returned chunk belongs to a document the principal has been allowed to see. Checked on
    every query, settled or not. If the response epoch covers the last ACL change, the current
    ACL is enforced strictly."""
    out: list[Violation] = []
    for h in result.hits:
        st = truth.docs.get(h.doc_id)
        if st is None or principal not in st.ever_allowed:
            out.append(Violation("I4", f"{principal} was never allowed {h.doc_id}"))
        elif (
            st.live
            and principal not in st.allowed
            and result.epoch is not None
            and st.acl_ack is not None
            and st.acl_ack.epoch is not None
            and result.epoch >= st.acl_ack.epoch
        ):
            out.append(Violation("I4", f"{principal} no longer allowed {h.doc_id} at epoch {result.epoch}"))
    return out[:1]


def check_i5(result: Result) -> list[Violation]:
    """Within one response each document appears in exactly one version."""
    seen: dict[str, int] = {}
    for h in result.hits:
        v = hit_version(h)
        if v is None:
            continue
        if seen.setdefault(h.doc_id, v) != v:
            return [Violation("I5", f"{h.doc_id} returned in versions {seen[h.doc_id]} and {v}")]
    return []


def check_presence(doc_id: str, version: int, result: Result) -> list[Violation]:
    if any(h.doc_id == doc_id and hit_version(h) == version for h in result.hits):
        return []
    return [Violation("PRESENCE", f"{doc_id} v{version} not returned for its own text")]
