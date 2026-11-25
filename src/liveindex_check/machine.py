"""Hypothesis stateful test: random upserts, updates, deletes and ACL changes interleaved with queries."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, precondition, rule, run_state_machine_as_test

from liveindex_check.adapter import Ack, Adapter, Result
from liveindex_check.docgen import make_text
from liveindex_check.invariants import (
    DocState,
    Truth,
    Violation,
    check_i1,
    check_i2_new,
    check_i2_old,
    check_i3,
    check_i4,
    check_i5,
    check_presence,
)

PRINCIPALS = ["u1", "u2", "u3"]
K = 10


@dataclass
class Timing:
    """Deadlines: how long to wait for a write to settle and how long to poll for invisibility."""

    settle_s: float = 5.0
    tti_s: float = 5.0
    poll_s: float = 0.02


TIMING = Timing()


@dataclass
class Stats:
    ops: int = 0
    queries: int = 0
    violations: dict[str, int] = field(default_factory=dict)
    tti: dict[str, list[float]] = field(default_factory=lambda: {"delete": [], "revoke": [], "update": []})
    examples: int = 0
    samples: list[dict] = field(default_factory=list)  # first few violations, for the report


@dataclass
class FailureHolder:
    steps: list[dict] = field(default_factory=list)
    violation: Violation | None = None


class SuiteViolation(AssertionError):
    pass


def _pick(seq: list, i: int):
    return seq[i % len(seq)]


def build_machine(adapter: Adapter, stats: Stats, holder: FailureHolder, raise_on_violation: bool):
    class Machine(RuleBasedStateMachine):
        def __init__(self) -> None:
            super().__init__()
            adapter.reset()
            stats.examples += 1
            self.truth = Truth()
            self.last_ack: Ack | None = None
            self.steps: list[dict] = []
            self.n_docs = 0

        # ---- plumbing ----
        def _record(self, **step) -> None:
            self.steps.append(step)
            holder.steps = list(self.steps)

        def _violate(self, vs: list[Violation]) -> None:
            for v in vs:
                stats.violations[v.invariant] = stats.violations.get(v.invariant, 0) + 1
                if len(stats.samples) < 20:
                    stats.samples.append({"invariant": v.invariant, "detail": v.detail, "step": len(self.steps)})
                holder.violation = v
                holder.steps = list(self.steps)
            if vs and raise_on_violation:
                raise SuiteViolation(f"{vs[0].invariant}: {vs[0].detail}")

        def _search(self, principal: str, query: str, settle: bool) -> Result:
            if settle and self.last_ack is not None:
                adapter.settle(self.last_ack, TIMING.settle_s)
            res = adapter.search(principal, query, K, self.last_ack if settle else None)
            stats.queries += 1
            self._violate(check_i4(self.truth, principal, res) + check_i5(res))
            return res

        def _write(self, ack: Ack) -> None:
            stats.ops += 1
            self.last_ack = ack

        def _tti(self, kind: str, ack: Ack, principal: str, query: str, check) -> None:
            """Settle, query once (the invariant check), then poll until clean to time it."""
            t0 = time.perf_counter()
            adapter.settle(ack, TIMING.settle_s)
            first = True
            while True:
                res = adapter.search(principal, query, K, ack)
                stats.queries += 1
                bad = check(res)
                if first:
                    self._violate(check_i4(self.truth, principal, res) + bad)
                    first = False
                if not bad:
                    stats.tti[kind].append(time.perf_counter() - t0)
                    return
                if time.perf_counter() - t0 > TIMING.tti_s:
                    stats.tti[kind].append(TIMING.tti_s)
                    return
                time.sleep(TIMING.poll_s)

        def _live_with_acl(self) -> list[str]:
            return [d for d in self.truth.live_docs() if self.truth.docs[d].allowed]

        # ---- writes ----
        @rule(allowed=st.sets(st.sampled_from(PRINCIPALS)), nonce=st.integers(0, 10**6), tenant=st.sampled_from(["t1", "t2"]))
        def upsert_new(self, allowed, nonce, tenant) -> None:
            doc_id = f"d{self.n_docs}"
            self.n_docs += 1
            text = make_text(doc_id, 1, nonce)
            self._record(op="upsert_new", doc=doc_id, version=1, allowed=sorted(allowed))
            ack = adapter.upsert(doc_id, text, set(allowed), tenant)
            st_ = DocState(version=1, texts={1: text}, live=True, allowed=set(allowed), ever_allowed=set(allowed), acl_ack=ack)
            self.truth.docs[doc_id] = st_
            self._write(ack)

        @precondition(lambda self: bool(self.truth.live_docs()))
        @rule(i=st.integers(0, 10**6), nonce=st.integers(0, 10**6))
        def upsert_update(self, i, nonce) -> None:
            doc_id = _pick(self.truth.live_docs(), i)
            d = self.truth.docs[doc_id]
            old_v, old_text = d.version, d.texts[d.version]
            new_v = old_v + 1
            text = make_text(doc_id, new_v, nonce)
            self._record(op="upsert_update", doc=doc_id, version=new_v, allowed=sorted(d.allowed))
            ack = adapter.upsert(doc_id, text, set(d.allowed), "t1")
            d.version = new_v
            d.texts[new_v] = text
            self.truth.tombstones.append(("old_version", doc_id, old_v, None, old_text))
            self._write(ack)
            principals = sorted(d.allowed)
            if principals:
                p = principals[0]

                def check(res: Result, doc_id=doc_id, old_v=old_v, new_v=new_v):
                    # old text must be gone and the new text found; query the old text for staleness
                    return check_i2_old(doc_id, old_v, new_v, res)

                self._tti("update", ack, p, old_text, check)
                res = self._search(p, text, settle=True)
                self._violate(check_i2_new(doc_id, new_v, res))

        @precondition(lambda self: bool(self.truth.live_docs()))
        @rule(i=st.integers(0, 10**6))
        def delete(self, i) -> None:
            doc_id = _pick(self.truth.live_docs(), i)
            d = self.truth.docs[doc_id]
            text = d.texts[d.version]
            self._record(op="delete", doc=doc_id)
            ack = adapter.delete(doc_id)
            d.live = False
            d.acl_ack = ack
            self.truth.tombstones.append(("deleted", doc_id, d.version, None, text))
            self._write(ack)
            principals = sorted(d.allowed) or sorted(d.ever_allowed)
            if principals:
                self._tti("delete", ack, principals[0], text, lambda res: check_i1(doc_id, res))

        @precondition(lambda self: bool(self._live_with_acl()))
        @rule(i=st.integers(0, 10**6), j=st.integers(0, 10**6))
        def revoke(self, i, j) -> None:
            doc_id = _pick(self._live_with_acl(), i)
            d = self.truth.docs[doc_id]
            p = _pick(sorted(d.allowed), j)
            text = d.texts[d.version]
            self._record(op="revoke", doc=doc_id, principal=p)
            d.allowed.discard(p)
            ack = adapter.set_acl(doc_id, set(d.allowed))
            d.acl_ack = ack
            self.truth.tombstones.append(("revoked", doc_id, d.version, p, text))
            self._write(ack)
            self._tti("revoke", ack, p, text, lambda res: check_i3(doc_id, res))

        @precondition(lambda self: bool(self.truth.live_docs()))
        @rule(i=st.integers(0, 10**6), j=st.integers(0, len(PRINCIPALS) - 1))
        def grant(self, i, j) -> None:
            doc_id = _pick(self.truth.live_docs(), i)
            d = self.truth.docs[doc_id]
            p = PRINCIPALS[j]
            self._record(op="grant", doc=doc_id, principal=p)
            d.allowed.add(p)
            d.ever_allowed.add(p)
            ack = adapter.set_acl(doc_id, set(d.allowed))
            d.acl_ack = ack
            self._write(ack)

        # ---- queries ----
        @precondition(lambda self: bool(self._live_with_acl()))
        @rule(i=st.integers(0, 10**6), j=st.integers(0, 10**6))
        def search_present(self, i, j) -> None:
            doc_id = _pick(self._live_with_acl(), i)
            d = self.truth.docs[doc_id]
            p = _pick(sorted(d.allowed), j)
            self._record(op="search_present", doc=doc_id, principal=p)
            res = self._search(p, d.texts[d.version], settle=True)
            self._violate(check_presence(doc_id, d.version, res))

        @precondition(lambda self: bool(self.truth.tombstones))
        @rule(i=st.integers(0, 10**6), j=st.integers(0, 10**6))
        def search_absent(self, i, j) -> None:
            kind, doc_id, version, principal, text = _pick(self.truth.tombstones, i)
            d = self.truth.docs[doc_id]
            p = principal or _pick(sorted(d.ever_allowed or set(PRINCIPALS)), j)
            self._record(op="search_absent", kind=kind, doc=doc_id, version=version, principal=p)
            res = self._search(p, text, settle=True)
            if kind == "deleted":
                self._violate(check_i1(doc_id, res))
            elif kind == "revoked":
                if p not in d.allowed:
                    self._violate(check_i3(doc_id, res))
            else:
                self._violate(check_i2_old(doc_id, version, d.version, res))

        @precondition(lambda self: bool(self.truth.docs))
        @rule(i=st.integers(0, 10**6), j=st.integers(0, len(PRINCIPALS) - 1))
        def search_unsettled(self, i, j) -> None:
            doc_id = _pick(sorted(self.truth.docs), i)
            d = self.truth.docs[doc_id]
            p = PRINCIPALS[j]
            self._record(op="search_unsettled", doc=doc_id, principal=p)
            self._search(p, d.texts[d.version], settle=False)

    return Machine


def run_machine(
    adapter: Adapter,
    stats: Stats,
    holder: FailureHolder,
    *,
    max_examples: int,
    steps: int,
    raise_on_violation: bool,
) -> None:
    cfg = settings(
        max_examples=max_examples,
        stateful_step_count=steps,
        deadline=None,
        database=None,
        derandomize=False,
        print_blob=False,
        suppress_health_check=list(HealthCheck),
    )
    run_state_machine_as_test(build_machine(adapter, stats, holder, raise_on_violation), settings=cfg)
