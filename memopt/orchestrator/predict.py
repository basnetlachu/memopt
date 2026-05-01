"""Predictor (orchestrator v1 §2.3.2, DECISION 6).

Tenant-keyed Markov + sequential + recency predictor. Greenfield —
does NOT import from memopt.vmm._oracle_py or memopt.vmm.oracle. The
algorithmic shape mirrors _oracle_py.py for parity (transition /
sequential / recency / fallback), but every key is `(tenant, tag)` so
two tenants observing the same tag values produce disjoint state
(G3).
"""
from __future__ import annotations

import collections
import re
import threading
from dataclasses import dataclass
from typing import Counter, Dict, List, Tuple


_TenantTag = Tuple[str, str]
_SEQ_RE = re.compile(r"^(.*?)(\d+)$")


@dataclass(frozen=True)
class Prediction:
    tenant: str
    tag: str
    confidence: float
    source: str  # "transition" | "sequential" | "recency" | "fallback"


class Predictor:
    """Per-tenant first-order Markov chain over allocation tags."""

    def __init__(
        self,
        max_transitions: int = 100_000,
        min_confidence: float = 0.3,
    ) -> None:
        # OrderedDict so we can LRU-evict the least-recently-touched row.
        self._transitions: "collections.OrderedDict[_TenantTag, Counter[_TenantTag]]" = (
            collections.OrderedDict()
        )
        self._recency: "collections.OrderedDict[_TenantTag, int]" = (
            collections.OrderedDict()
        )
        self._steps: Dict[str, int] = {}
        self._last_tag: Dict[str, str] = {}
        self._max_transitions = max_transitions
        self._min_confidence = min_confidence
        self._lock = threading.RLock()

    def observe(self, tenant: str, tag: str) -> None:
        with self._lock:
            prev = self._last_tag.get(tenant)
            if prev is not None:
                key = (tenant, prev)
                cnt = self._transitions.get(key)
                if cnt is None:
                    cnt = collections.Counter()
                    self._transitions[key] = cnt
                else:
                    self._transitions.move_to_end(key)
                cnt[(tenant, tag)] += 1
                # Bound the table by row count.
                while len(self._transitions) > self._max_transitions:
                    self._transitions.popitem(last=False)
            self._last_tag[tenant] = tag
            self._steps[tenant] = self._steps.get(tenant, 0) + 1
            rkey = (tenant, tag)
            if rkey in self._recency:
                del self._recency[rkey]
            self._recency[rkey] = self._steps[tenant]

    def predict(
        self, tenant: str, tag: str, top_k: int = 10
    ) -> List[Prediction]:
        with self._lock:
            results: Dict[str, Prediction] = {}

            # Source 1 — transition.
            cnt = self._transitions.get((tenant, tag))
            if cnt:
                total = sum(cnt.values())
                for (t2, next_tag), c in cnt.most_common(top_k):
                    conf = c / total
                    if conf >= self._min_confidence:
                        results[next_tag] = Prediction(
                            tenant=t2,
                            tag=next_tag,
                            confidence=conf,
                            source="transition",
                        )

            # Source 2 — sequential (numeric suffix +1 / +2).
            m = _SEQ_RE.match(tag)
            if m:
                prefix, num_str = m.group(1), m.group(2)
                width = len(num_str)
                base = int(num_str)
                for delta, conf in ((1, 0.6), (2, 0.4)):
                    if conf < self._min_confidence:
                        continue
                    nxt_num = str(base + delta).zfill(width)
                    nxt = f"{prefix}{nxt_num}"
                    if nxt not in results:
                        results[nxt] = Prediction(
                            tenant=tenant,
                            tag=nxt,
                            confidence=conf,
                            source="sequential",
                        )

            # Source 3 — recency (other recent tags for this tenant).
            recent = [
                (k[1], v)
                for k, v in self._recency.items()
                if k[0] == tenant and k[1] != tag
            ]
            recent.sort(key=lambda x: x[1], reverse=True)
            recency_conf = 0.35
            for nxt, _step in recent[:5]:
                if nxt in results:
                    continue
                if recency_conf < self._min_confidence:
                    break
                results[nxt] = Prediction(
                    tenant=tenant,
                    tag=nxt,
                    confidence=recency_conf,
                    source="recency",
                )

            # Source 4 — fallback (cold-start safety net).
            fallback_conf = 0.3
            if not results and fallback_conf >= self._min_confidence:
                results[tag] = Prediction(
                    tenant=tenant,
                    tag=tag,
                    confidence=fallback_conf,
                    source="fallback",
                )

            return sorted(
                results.values(),
                key=lambda p: p.confidence,
                reverse=True,
            )[:top_k]

    def forget_tenant(self, tenant: str) -> None:
        with self._lock:
            stale_tx = [k for k in self._transitions if k[0] == tenant]
            for k in stale_tx:
                self._transitions.pop(k, None)
            # Also drop transitions whose value-side counter only ever pointed at this tenant.
            for k, cnt in list(self._transitions.items()):
                drop = [tk for tk in cnt if tk[0] == tenant]
                for tk in drop:
                    del cnt[tk]
                if not cnt:
                    self._transitions.pop(k, None)
            stale_rec = [k for k in self._recency if k[0] == tenant]
            for k in stale_rec:
                self._recency.pop(k, None)
            self._steps.pop(tenant, None)
            self._last_tag.pop(tenant, None)

    def stats(self) -> dict:
        with self._lock:
            return {
                "transition_rows": len(self._transitions),
                "transition_cells": sum(
                    len(c) for c in self._transitions.values()
                ),
                "recency_size": len(self._recency),
                "tenants_tracked": len(self._steps),
                "max_transitions": self._max_transitions,
                "min_confidence": self._min_confidence,
            }
