"""Epoch-versioned ACL state: per-principal grants and group membership, in memory."""
from __future__ import annotations

import bisect

import numpy as np

INF = 9223372036854775807


class AclState:
    """Single writer (the indexer). Readers only call `allowed_docs`; entries are append-only
    and a revoke only rewrites `revoked` with the revoking epoch, so an older epoch still sees
    the old grant."""

    def __init__(self) -> None:
        # principal -> (doc_ix list, granted list, revoked list)
        self.grants: dict[str, tuple[list[int], list[int], list[int]]] = {}
        # group -> member -> list of [added, removed]
        self.members: dict[str, dict[str, list[list[int]]]] = {}
        self.acl_epochs: list[int] = []
        self._cache: dict[tuple[str, int, int], np.ndarray] = {}

    def _touch(self, epoch: int) -> None:
        if not self.acl_epochs or self.acl_epochs[-1] != epoch:
            self.acl_epochs.append(epoch)

    def grant(self, doc_ix: int, principal: str, epoch: int) -> None:
        d, g, r = self.grants.setdefault(principal, ([], [], []))
        d.append(doc_ix)
        g.append(epoch)
        r.append(INF)
        self._touch(epoch)

    def revoke(self, doc_ix: int, principal: str, epoch: int) -> None:
        d, _, r = self.grants.get(principal, ([], [], []))
        for i in range(len(d) - 1, -1, -1):
            if d[i] == doc_ix and r[i] == INF:
                r[i] = epoch
                break
        self._touch(epoch)

    def revoke_all(self, doc_ix: int, epoch: int) -> None:
        for d, _, r in self.grants.values():
            for i, di in enumerate(d):
                if di == doc_ix and r[i] == INF:
                    r[i] = epoch
        self._touch(epoch)

    def group_add(self, grp: str, member: str, epoch: int) -> None:
        self.members.setdefault(grp, {}).setdefault(member, []).append([epoch, INF])
        self._touch(epoch)

    def group_remove(self, grp: str, member: str, epoch: int) -> None:
        spans = self.members.get(grp, {}).get(member, [])
        if spans and spans[-1][1] == INF:
            spans[-1][1] = epoch
        self._touch(epoch)

    def acl_epoch(self, epoch: int) -> int:
        i = bisect.bisect_right(self.acl_epochs, epoch)
        return self.acl_epochs[i - 1] if i else 0

    def principals_at(self, principal: str, epoch: int) -> list[str]:
        out = [principal]
        for grp, mem in list(self.members.items()):
            if any(a <= epoch < b for a, b in mem.get(principal, [])):
                out.append(grp)
        return out

    def allowed_docs(self, principal: str, epoch: int, n_docs: int) -> np.ndarray:
        """Boolean mask over documents allowed for `principal` at `epoch`. Cached per ACL epoch."""
        key = (principal, self.acl_epoch(epoch), n_docs)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        mask = np.zeros(n_docs, dtype=bool)
        for p in self.principals_at(principal, epoch):
            entry = self.grants.get(p)
            if not entry:
                continue
            k = min(len(entry[0]), len(entry[1]), len(entry[2]))
            d = np.asarray(entry[0][:k], dtype=np.int64)
            g = np.asarray(entry[1][:k], dtype=np.int64)
            r = np.asarray(entry[2][:k], dtype=np.int64)
            sel = (g <= epoch) & (r > epoch) & (d < n_docs)
            mask[d[sel]] = True
        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[key] = mask
        return mask
