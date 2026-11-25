"""In-memory fake systems for testing the suite itself. `bug` selects one deliberate defect."""
from __future__ import annotations

import re

from liveindex_check.adapter import Ack, Hit, Result

MARK = re.compile(r"zq-[a-z0-9]+-\d+-\d+")


class FakeAdapter:
    def __init__(self, bug: str | None = None):
        self.bug = bug
        self.reset()

    def reset(self) -> None:
        self.docs: dict[str, dict] = {}
        self.n = 0

    def _ack(self) -> Ack:
        self.n += 1
        return Ack(str(self.n), None)

    def upsert(self, doc_id: str, text: str, allowed: set[str], tenant: str) -> Ack:
        d = self.docs.get(doc_id)
        if d is None:
            d = self.docs[doc_id] = {"versions": [], "live": True, "allowed": set(), "ever": set(), "sticky": set()}
        version = d.get("version_no", 0) + 1
        if self.bug == "I2":
            d["versions"].append((version, text))  # old versions stay searchable
        else:
            d["versions"] = [(version, text)]
        d["version_no"] = version
        d["live"] = True
        self._acl(d, allowed)
        return self._ack()

    def _acl(self, d: dict, allowed: set[str]) -> None:
        d["allowed"] = set(allowed)
        d["ever"] |= allowed
        d["sticky"] |= allowed

    def delete(self, doc_id: str) -> Ack:
        if self.bug != "I1":
            self.docs[doc_id]["live"] = False
        return self._ack()

    def set_acl(self, doc_id: str, allowed: set[str]) -> Ack:
        self._acl(self.docs[doc_id], allowed)
        return self._ack()

    def search(self, principal: str, query: str, k: int, min_ack: Ack | None) -> Result:
        m = MARK.search(query)
        hits: list[Hit] = []
        if m:
            for doc_id, d in self.docs.items():
                if not d["live"]:
                    continue
                if self.bug == "I3":
                    ok = principal in d["sticky"]
                elif self.bug == "I4":
                    ok = principal in d["allowed"] or principal not in d["ever"]
                else:
                    ok = principal in d["allowed"]
                if not ok:
                    continue
                for v, text in d["versions"]:
                    if m.group(0) in text:
                        hits.append(Hit(doc_id, v, text))
        return Result(hits[:k], None)

    def settle(self, ack: Ack, timeout_s: float) -> bool:
        return True
