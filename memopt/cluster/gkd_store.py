"""
Global KV Cache Deduplication Store (GKD).

The GKD store is the economic core of the Distributed Memory Hypervisor.
It answers one question: "Has any node in the cluster already computed
the KV cache for this exact prompt prefix?"

If yes  → return a pointer to the existing block. Zero recompute. Zero HBM.
If no   → compute normally, store the result, register the hash.

Architecture:
    GKDStore
    ├── HashEngine          — SHA-256 + collision verification
    ├── LocalGKDBackend     — dict-based, single-node, no dependencies
    └── RedisGKDBackend     — Redis-based, cluster-wide, production

Integration with VMM:
    When the VMM allocates a new KV block, it calls:
        hit = gkd.lookup(token_ids, sequence_length)
        if hit:
            return hit.block_ref   # skip allocation entirely
        else:
            block = vmm.allocate(...)
            gkd.register(token_ids, sequence_length, block_ref)

Instrumentation (critical for design partner demos):
    gkd.stats() returns:
        - total_lookups
        - cache_hits
        - cache_misses
        - hit_rate_pct
        - estimated_hbm_saved_gb   ← the number that sells the product
        - estimated_compute_saved_pct
        - collision_checks_total
        - collision_detections_total   ← must always be 0
"""
from __future__ import annotations
import time
import logging
import threading
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from .hashing import compute_hash, make_fingerprint, verify_fingerprint

logger = logging.getLogger(__name__)


@dataclass
class GKDEntry:
    """One entry in the dedup store."""
    content_hash:  str
    fingerprint:   List[int]      # first 64 tokens for collision check
    block_ref:     str            # VMM block identifier (sequence_id:block_index)
    node_id:       str            # which node holds the physical block
    size_bytes:    int            # size of the cached KV block
    registered_at: float = field(default_factory=time.monotonic)
    hit_count:     int = 0        # how many times this entry was served


@dataclass
class GKDHit:
    """Returned by lookup() on a cache hit."""
    block_ref:   str
    node_id:     str
    size_bytes:  int           = 0
    hit_count:   int           = 0
    matched_len: Optional[int] = None   # tokens matched (None = full)
    delta_start: Optional[int] = None   # recompute from here
    is_partial:  bool          = False  # True = LCP match


class LocalGKDBackend:
    """
    In-memory backend for single-node development and testing.

    Thread-safe via RLock. No external dependencies.
    All data is lost when the process exits — this is intentional.
    Production deployments use RedisGKDBackend for persistence and
    cross-node sharing.
    """

    def __init__(self):
        self._store: Dict[str, dict] = {}
        self._lock  = threading.RLock()

    def get(self, content_hash: str) -> Optional[dict]:
        with self._lock:
            return self._store.get(content_hash)

    def set(self, content_hash: str, entry: dict, ttl_seconds: int = 3600):
        """TTL is ignored in the local backend — entries live until evicted or cleared."""
        with self._lock:
            self._store[content_hash] = entry

    def increment_hit(self, content_hash: str):
        with self._lock:
            if content_hash in self._store:
                self._store[content_hash]["hit_count"] += 1

    def delete(self, content_hash: str):
        with self._lock:
            self._store.pop(content_hash, None)

    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def clear(self):
        with self._lock:
            self._store.clear()


class RedisGKDBackend:
    """
    Redis-backed backend for cluster-wide deduplication.

    All nodes in the cluster connect to the same Redis instance.
    A KV block computed on Node A is immediately available to Node B.

    Redis key format:  gkd:v1:{content_hash}
    Redis value:       JSON-serialised GKDEntry fields
    TTL:               Configurable, default 1 hour. Entries expire
                       automatically — no manual eviction needed.

    Failure mode:
        If Redis is unreachable, all operations log a warning and return
        None (treated as cache miss). The system degrades gracefully to
        no deduplication — it never crashes inference.
    """

    _KEY_PREFIX = "gkd:v1:"

    def __init__(self, host: str = "localhost", port: int = 6379,
                 db: int = 0, password: Optional[str] = None,
                 socket_timeout: float = 0.005):
        """
        Args:
            host:             Redis hostname or IP.
            port:             Redis port (default 6379).
            db:               Redis DB index (default 0).
            password:         Redis AUTH password, if set.
            socket_timeout:   Max wait per Redis operation in seconds.
                              0.005 = 5ms. Keeps GKD lookup off the
                              hot inference path even if Redis is slow.
        """
        self._degraded       = False
        self._degraded_since: Optional[float] = None

        try:
            import redis, json
            self._redis = redis.Redis(
                host=host, port=port, db=db,
                password=password,
                socket_timeout=socket_timeout,
                decode_responses=True,
            )
            self._json = json
            self._available = True
            self._redis.ping()
            logger.info("GKD Redis backend connected: %s:%d", host, port)
        except Exception as e:
            self._available = False
            self._local_fallback = LocalGKDBackend()
            self._mark_degraded(str(e))

    def _mark_degraded(self, reason: str) -> None:
        """Mark this backend as degraded and log at ERROR level once."""
        if not self._degraded:
            self._degraded       = True
            self._degraded_since = time.monotonic()
            logger.error(
                "GKD Redis DEGRADED: %s. "
                "Falling back to local backend — cluster-wide "
                "deduplication is disabled. "
                "Check Redis connectivity and set REDIS_URL.",
                reason,
            )

    @property
    def degraded(self) -> bool:
        return self._degraded

    @property
    def degraded_since(self) -> Optional[float]:
        return self._degraded_since

    def _key(self, content_hash: str) -> str:
        return f"{self._KEY_PREFIX}{content_hash}"

    def get(self, content_hash: str) -> Optional[dict]:
        if not self._available:
            return self._local_fallback.get(content_hash)
        try:
            raw = self._redis.get(self._key(content_hash))
            return self._json.loads(raw) if raw else None
        except Exception as e:
            logger.debug("GKD Redis GET failed: %s", e)
            return None

    def set(self, content_hash: str, entry: dict, ttl_seconds: int = 3600):
        if not self._available:
            return self._local_fallback.set(content_hash, entry, ttl_seconds)
        try:
            self._redis.setex(
                self._key(content_hash),
                ttl_seconds,
                self._json.dumps(entry),
            )
        except Exception as e:
            logger.debug("GKD Redis SET failed: %s", e)

    def increment_hit(self, content_hash: str):
        if not self._available:
            return self._local_fallback.increment_hit(content_hash)
        try:
            self._redis.hincrby(self._key(content_hash), "hit_count", 1)
        except Exception as e:
            logger.debug("GKD Redis HINCRBY failed: %s", e)

    def delete(self, content_hash: str):
        if not self._available:
            return self._local_fallback.delete(content_hash)
        try:
            self._redis.delete(self._key(content_hash))
        except Exception as e:
            logger.debug("GKD Redis DELETE failed: %s", e)

    def size(self) -> int:
        if not self._available:
            return self._local_fallback.size()
        try:
            cursor, keys = self._redis.scan(0, match=f"{self._KEY_PREFIX}*", count=1000)
            return len(keys)
        except Exception:
            return -1

    def clear(self):
        if not self._available:
            return self._local_fallback.clear()
        try:
            cursor = 0
            while True:
                cursor, keys = self._redis.scan(cursor, match=f"{self._KEY_PREFIX}*", count=500)
                if keys:
                    self._redis.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.debug("GKD Redis CLEAR failed: %s", e)


class GKDStore:
    """
    Global KV Cache Deduplication Store.

    Primary API for the VMM and serving layer.
    Thread-safe. Backend-agnostic (local dict or Redis cluster).

    Usage:
        # Single-node development
        gkd = GKDStore()

        # Cluster production
        gkd = GKDStore(backend="redis", redis_host="redis.internal")

        # In VMM allocation path:
        hit = gkd.lookup(token_ids=[101, 202, 303, ...], sequence_length=512)
        if hit:
            # Skip KV computation entirely — serve from hit.block_ref
            pass
        else:
            block = vmm.allocate(seq_id, block_index, size_bytes)
            gkd.register(
                token_ids=token_ids,
                sequence_length=512,
                block_ref=f"{seq_id}:{block_index}",
                node_id="node-a",
                size_bytes=block_size,
            )
    """

    def __init__(
        self,
        backend: str = "local",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        default_ttl_seconds: int = 3600,
        block_size_bytes: int = 131_072,
        redis_url: Optional[str] = None,
        node_id: str = "local",
    ):
        """
        Args:
            backend:             "local" (default, no deps) or "redis" (cluster).
            redis_host:          Redis hostname. Used only when backend="redis".
            redis_port:          Redis port.
            redis_password:      Redis AUTH password.
            default_ttl_seconds: How long entries live without being accessed.
                                 1 hour default. Popular prompts stay hot.
            block_size_bytes:    Default KV block size in bytes.
                                 Used to estimate HBM saved in stats().
            redis_url:           Optional redis:// or rediss:// URL. When provided,
                                 overrides redis_host/redis_port/redis_password and
                                 forces backend="redis".
            node_id:             Identifier for this node in stats and logs.
        """
        self._node_id = node_id

        if redis_url is not None:
            # Parse redis[s]://[:password@]host[:port][/db]
            parsed       = urllib.parse.urlparse(redis_url)
            redis_host   = parsed.hostname or "localhost"
            redis_port   = parsed.port or 6379
            redis_password = parsed.password or redis_password
            backend      = "redis"

        if backend == "redis":
            self._backend = RedisGKDBackend(
                host=redis_host, port=redis_port, password=redis_password
            )
        else:
            self._backend = LocalGKDBackend()

        self._ttl              = default_ttl_seconds
        self._block_size_bytes = block_size_bytes

        self._lock                 = threading.Lock()
        self._total_lookups        = 0
        self._cache_hits           = 0
        self._cache_misses         = 0
        self._bytes_saved          = 0
        self._collision_checks     = 0
        self._collision_detections = 0
        self._exact_hits:        int = 0
        self._lcp_hits:          int = 0
        self._lcp_tokens_reused: int = 0
        self._lcp_tokens_total:  int = 0

    # ── Primary API ────────────────────────────────────────────────────

    def lookup(
        self,
        token_ids: List[int],
        sequence_length: int,
    ) -> Optional[GKDHit]:
        """
        Look up whether a KV block for this token sequence already exists.

        Args:
            token_ids:        The prompt token IDs (prefix that drives KV cache).
            sequence_length:  Total length — used in hash to prevent prefix
                              collisions.

        Returns:
            GKDHit  if a verified cache entry exists — caller should skip
                    KV computation and use hit.block_ref directly.
            None    if no entry exists (cache miss) — caller should compute
                    normally then call register().

        Performance:
            Local backend: ~0.01ms
            Redis backend: ~0.5–2ms (network RTT to Redis)
        """
        with self._lock:
            self._total_lookups += 1

        content_hash = compute_hash(token_ids, sequence_length)
        raw = self._backend.get(content_hash)

        if raw is None:
            # LCP fallback — only reached on exact miss
            try:
                from memopt.cluster.prefix_index import lookup_longest_prefix
                result = lookup_longest_prefix(
                    token_ids, sequence_length, self._backend
                )
                if result is not None:
                    block_ref, node_id, matched_len = result
                    with self._lock:
                        self._cache_hits        += 1
                        self._lcp_hits          += 1
                        self._lcp_tokens_reused += matched_len
                        self._lcp_tokens_total  += sequence_length
                    return GKDHit(
                        block_ref   = block_ref,
                        node_id     = node_id,
                        hit_count   = 1,
                        matched_len = matched_len,
                        delta_start = matched_len,
                        is_partial  = True,
                    )
            except Exception as exc:
                logger.debug("GKD LCP lookup failed: %s", exc)

            with self._lock:
                self._cache_misses += 1
            return None

        # Collision verification — the critical safety check
        with self._lock:
            self._collision_checks += 1

        stored_fingerprint = raw.get("fingerprint", [])
        if not verify_fingerprint(stored_fingerprint, token_ids):
            with self._lock:
                self._collision_detections += 1
            logger.error(
                "GKD COLLISION DETECTED for hash %s... "
                "Treating as miss. Investigate immediately.",
                content_hash[:16],
            )
            return None

        # Genuine hit
        with self._lock:
            self._cache_hits += 1
            self._exact_hits += 1
            self._bytes_saved += raw.get("size_bytes", self._block_size_bytes)

        self._backend.increment_hit(content_hash)

        return GKDHit(
            block_ref  = raw["block_ref"],
            node_id    = raw["node_id"],
            size_bytes = raw.get("size_bytes", self._block_size_bytes),
            hit_count  = raw.get("hit_count", 0) + 1,
        )

    def register(
        self,
        token_ids: List[int],
        sequence_length: int,
        block_ref: str,
        node_id: str,
        size_bytes: Optional[int] = None,
        ttl_seconds: Optional[int] = None,
    ):
        """
        Register a newly computed KV block in the dedup store.

        Call this after every cache miss once the KV block has been
        allocated in the VMM. Future lookups with the same token_ids
        will return a hit pointing to this block.

        Args:
            token_ids:       The prompt token IDs used to compute this block.
            sequence_length: Total sequence length.
            block_ref:       VMM block identifier: "{sequence_id}:{block_index}".
            node_id:         ID of the node holding the physical block.
            size_bytes:      Size of the KV block in bytes.
            ttl_seconds:     Override the store's default TTL for this entry.
        """
        content_hash = compute_hash(token_ids, sequence_length)
        fingerprint  = make_fingerprint(token_ids)

        entry = {
            "content_hash":  content_hash,
            "fingerprint":   fingerprint,
            "block_ref":     block_ref,
            "node_id":       node_id,
            "size_bytes":    size_bytes or self._block_size_bytes,
            "registered_at": time.time(),
            "hit_count":     0,
        }

        self._backend.set(content_hash, entry, ttl_seconds=ttl_seconds or self._ttl)
        logger.debug("GKD registered: hash=%s... node=%s ref=%s", content_hash[:16], node_id, block_ref)

        try:
            from memopt.cluster.prefix_index import register_prefixes
            register_prefixes(
                token_ids, sequence_length, block_ref, node_id, self._backend
            )
        except Exception as exc:
            logger.debug("GKD prefix registration failed: %s", exc)

    def invalidate(self, token_ids: List[int], sequence_length: int):
        """
        Remove an entry from the store.

        Call this when a KV block is evicted from all tiers and the
        block_ref is no longer valid. Prevents stale pointers.
        """
        content_hash = compute_hash(token_ids, sequence_length)
        self._backend.delete(content_hash)

    # ── Instrumentation ────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """
        Return instrumentation metrics.

        The 'estimated_hbm_saved_gb' number is what sells the product.

        Example output on a popular deployment:
            {
                "total_lookups":              50_000,
                "cache_hits":                 47_500,
                "cache_misses":                2_500,
                "hit_rate_pct":                 95.0,
                "estimated_hbm_saved_gb":      593.75,
                "estimated_compute_saved_pct":  95.0,
                "entries_in_store":             2_500,
                "collision_checks_total":      47_500,
                "collision_detections_total":       0,  ← must always be 0
            }
        """
        with self._lock:
            total       = self._total_lookups
            hits        = self._cache_hits
            misses      = self._cache_misses
            bytes_saved = self._bytes_saved
            col_checks  = self._collision_checks
            col_dets    = self._collision_detections

        hit_rate = (hits / total * 100.0) if total > 0 else 0.0

        return {
            "total_lookups":               total,
            "cache_hits":                  hits,
            "cache_misses":                misses,
            "hit_rate_pct":                round(hit_rate, 2),
            "estimated_hbm_saved_gb":      round(bytes_saved / 1e9, 3),
            "estimated_compute_saved_pct": round(hit_rate, 2),
            "entries_in_store":            self._backend.size(),
            "collision_checks_total":      col_checks,
            "collision_detections_total":  col_dets,
            "backend_degraded":            getattr(self._backend, "degraded", False),
            "backend_degraded_since":      getattr(self._backend, "degraded_since", None),
            "exact_hits":                  self._exact_hits,
            "lcp_hits":                    self._lcp_hits,
            "lcp_token_reuse_pct":         round(
                self._lcp_tokens_reused / self._lcp_tokens_total * 100, 1
            ) if self._lcp_tokens_total > 0 else 0.0,
            "total_hits":                  hits,
        }

    def reset_stats(self):
        """Reset all counters. Useful between benchmark runs."""
        with self._lock:
            self._total_lookups        = 0
            self._cache_hits           = 0
            self._cache_misses         = 0
            self._bytes_saved          = 0
            self._collision_checks     = 0
            self._collision_detections = 0
