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
    └── RedisGKDBackend     — Redis or Redis Cluster, production

Redis deployment options (set via REDIS_URL environment variable):
    Single instance:
        REDIS_URL=redis://localhost:6379
    Redis Cluster (3+ nodes, production):
        REDIS_URL=redis://node1:6379,redis://node2:6379,redis://node3:6379
    Redis Sentinel (HA single instance):
        REDIS_URL=redis+sentinel://sentinel1:26379/mymaster
"""
from __future__ import annotations
import json
import logging
import os
import time
import threading
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from .hashing import compute_hash, make_fingerprint, verify_fingerprint
from .prefix_index import _verify_fingerprint_fast

logger = logging.getLogger(__name__)


@dataclass
class GKDEntry:
    """One entry in the dedup store."""
    content_hash:  str
    fingerprint:   List[int]
    block_ref:     str
    node_id:       str
    size_bytes:    int
    registered_at: float = field(default_factory=time.monotonic)
    hit_count:     int = 0


@dataclass
class GKDHit:
    """Returned by lookup() on a cache hit."""
    block_ref:   str
    node_id:     str
    size_bytes:  int           = 0
    hit_count:   int           = 0
    matched_len: Optional[int] = None
    delta_start: Optional[int] = None
    is_partial:  bool          = False


class LocalGKDBackend:
    """
    In-memory backend for single-node development and testing.
    Thread-safe via RLock. No external dependencies.
    """

    def __init__(self):
        self._store: Dict[str, dict] = {}
        self._lock  = threading.RLock()

    def get(self, content_hash: str) -> Optional[dict]:
        with self._lock:
            return self._store.get(content_hash)

    def set(self, content_hash: str, entry, ttl_seconds: int = 3600):
        with self._lock:
            if isinstance(entry, str):
                self._store[content_hash] = entry
            else:
                self._store[content_hash] = entry

    def increment_hit(self, content_hash: str):
        with self._lock:
            e = self._store.get(content_hash)
            if e and isinstance(e, dict):
                e["hit_count"] = e.get("hit_count", 0) + 1

    def delete(self, content_hash: str):
        with self._lock:
            self._store.pop(content_hash, None)

    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def clear(self):
        with self._lock:
            self._store.clear()

    def pipeline_get(self, keys: List[str]) -> List[Optional[str]]:
        """Batch get — returns results in same order as keys."""
        with self._lock:
            return [self._store.get(k) for k in keys]


class RedisGKDBackend:
    """
    Redis-backed backend for cluster-wide deduplication.

    Supports both single Redis instance and Redis Cluster.
    Selection is automatic based on the URL format:
        Single:  redis://host:port
        Cluster: redis://host1:port1,redis://host2:port2,...

    Failure mode:
        If Redis is unreachable, all operations fall back to
        LocalGKDBackend. The system degrades gracefully —
        deduplication is disabled but inference never crashes.
    """

    _KEY_PREFIX = "gkd:v1:"

    def __init__(self, host: str = "localhost", port: int = 6379,
                 db: int = 0, password: Optional[str] = None,
                 socket_timeout: float = 0.005,
                 redis_url: Optional[str] = None):
        self._degraded       = False
        self._degraded_since: Optional[float] = None
        self._redis          = None
        self._cluster_mode   = False
        self._local_fallback = LocalGKDBackend()
        self._available      = False
        self._redis_url      = redis_url
        self._json           = json

        if redis_url:
            self._connect(redis_url)
        else:
            # Legacy host/port path
            try:
                import redis as redis_lib
                self._redis = redis_lib.Redis(
                    host=host, port=port, db=db,
                    password=password,
                    socket_timeout=socket_timeout,
                    decode_responses=True,
                )
                self._redis.ping()
                self._available = True
                self._cluster_mode = False
                logger.info("GKD Redis connected: %s:%d", host, port)
            except Exception as e:
                self._mark_degraded(str(e))

    def _connect(self, redis_url: str) -> None:
        """
        Connect to Redis. Tries Redis Cluster first if multiple URLs
        are provided, falls back to single instance.

        URL formats:
            Single:  redis://host:port
            Cluster: redis://host1:port1,redis://host2:port2,...
        """
        try:
            import redis as redis_lib
        except ImportError:
            self._mark_degraded("redis package not installed")
            return

        urls = [u.strip() for u in redis_url.split(",")]

        # ── Redis Cluster path (multiple URLs) ────────────────────────
        if len(urls) > 1:
            try:
                from redis.cluster import RedisCluster
            except ImportError:
                logger.warning(
                    "redis.cluster not available — "
                    "install redis>=4.1 for cluster support")
                # Fall through to single instance with first URL
                urls = [urls[0]]
            else:
                startup_nodes = []
                for url in urls:
                    parsed = urllib.parse.urlparse(url)
                    startup_nodes.append({
                        "host": parsed.hostname or "localhost",
                        "port": parsed.port or 6379,
                    })
                try:
                    client = RedisCluster(
                        startup_nodes=startup_nodes,
                        decode_responses=True,
                        skip_full_coverage_check=True,
                        retry_on_timeout=True,
                        socket_timeout=2.0,
                        socket_connect_timeout=2.0,
                    )
                    client.ping()
                    self._redis = client
                    self._cluster_mode = True
                    self._available = True
                    logger.info(
                        "GKD Redis Cluster connected (%d nodes)",
                        len(startup_nodes))
                    return
                except Exception as e:
                    logger.warning(
                        "GKD Redis Cluster failed: %s — "
                        "trying single instance", e)

        # ── Single instance path ──────────────────────────────────────
        try:
            client = redis_lib.Redis.from_url(
                urls[0],
                socket_timeout=2.0,
                socket_connect_timeout=2.0,
                retry_on_timeout=True,
                decode_responses=True,
            )
            client.ping()
            self._redis = client
            self._cluster_mode = False
            self._available = True
            logger.info("GKD Redis single instance connected")
        except Exception as e:
            self._mark_degraded(f"Redis unavailable: {e}")

    def _mark_degraded(self, reason: str) -> None:
        if not self._degraded:
            self._degraded       = True
            self._degraded_since = time.monotonic()
            logger.error(
                "GKD Redis DEGRADED: %s. "
                "Falling back to local backend — cluster-wide "
                "deduplication is disabled.",
                reason,
            )

    def _health_check(self) -> bool:
        """Called by MetricsCollector every 15s."""
        if self._redis is None:
            return False
        try:
            self._redis.ping()
            if self._degraded:
                logger.info("GKD Redis recovered")
                self._degraded = False
                self._degraded_since = None
            return True
        except Exception as e:
            self._mark_degraded(f"Redis ping failed: {e}")
            return False

    def reconnect(self) -> None:
        """Attempt to reconnect using the original URL."""
        if self._redis_url:
            self._connect(self._redis_url)

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

    def set(self, content_hash: str, entry, ttl_seconds: int = 3600):
        if not self._available:
            return self._local_fallback.set(content_hash, entry, ttl_seconds)
        try:
            val = entry if isinstance(entry, str) else self._json.dumps(entry)
            self._redis.setex(
                self._key(content_hash),
                ttl_seconds,
                val,
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
            cursor, keys = self._redis.scan(
                0, match=f"{self._KEY_PREFIX}*", count=1000)
            return len(keys)
        except Exception:
            return -1

    def clear(self):
        if not self._available:
            return self._local_fallback.clear()
        try:
            cursor = 0
            while True:
                cursor, keys = self._redis.scan(
                    cursor, match=f"{self._KEY_PREFIX}*", count=500)
                if keys:
                    self._redis.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.debug("GKD Redis CLEAR failed: %s", e)

    def pipeline_get(self, keys: List[str]) -> List[Optional[str]]:
        """
        Batch get using Redis pipeline — one round-trip for all keys.

        For Redis Cluster: uses pipeline(transaction=False) which
        works across cluster slots (non-atomic but correct for reads).
        For single Redis: standard pipeline.

        Returns results in same order as keys.
        Falls back to sequential gets on error.
        """
        if not self._available:
            return self._local_fallback.pipeline_get(keys)
        try:
            pipe = self._redis.pipeline(transaction=False)
            for k in keys:
                pipe.get(self._key(k))
            raw_results = pipe.execute()
            results = []
            for raw in raw_results:
                if raw is not None:
                    results.append(raw)
                else:
                    results.append(None)
            return results
        except Exception as e:
            logger.debug("GKD Redis pipeline failed: %s — "
                         "falling back to sequential", e)
            # Sequential fallback
            return [self.get(k) for k in keys]


class GKDStore:
    """
    Global KV Cache Deduplication Store.

    Primary API for the VMM and serving layer.
    Thread-safe. Backend-agnostic (local dict or Redis cluster).

    Usage:
        # Single-node development
        gkd = GKDStore()

        # Cluster production (single Redis)
        gkd = GKDStore(backend="redis", redis_host="redis.internal")

        # Cluster production (Redis Cluster)
        gkd = GKDStore(redis_url="redis://n1:6379,redis://n2:6379,redis://n3:6379")
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
        self._node_id = node_id

        if redis_url is not None:
            backend = "redis"

        if backend == "redis":
            if redis_url:
                self._backend = RedisGKDBackend(redis_url=redis_url)
            else:
                self._backend = RedisGKDBackend(
                    host=redis_host, port=redis_port,
                    password=redis_password)
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

        # Output cache: block_ref → completion output dict.
        # Used by exact-hit path in serving/server.py to skip inference.
        # Max 10K entries; evicts oldest on overflow.
        self._output_cache: Dict[str, dict] = {}
        self._output_cache_lock = threading.Lock()

        # Background TTL enforcement for LocalGKDBackend
        # (RedisGKDBackend uses native Redis EXPIRE)
        self._ttl_thread = threading.Thread(
            target=self._ttl_enforce_loop,
            daemon=True, name="gkd-ttl")
        self._ttl_thread.start()

    # ── TTL enforcement ───────────────────────────────────────────────

    def _ttl_enforce_loop(self) -> None:
        """Remove expired entries every 5 minutes."""
        while True:
            time.sleep(300)
            try:
                self._enforce_ttl()
            except Exception as e:
                logger.error(f"TTL enforcement error: {e}")

    def _enforce_ttl(self) -> None:
        """Remove entries older than MEMOPT_GKD_ENTRY_TTL_S from local backend."""
        ttl_s = float(os.environ.get('MEMOPT_GKD_ENTRY_TTL_S',
                                      str(self._ttl)))
        cutoff = time.time() - ttl_s
        removed = 0
        try:
            if hasattr(self._backend, '_store'):
                with self._backend._lock:
                    expired = [
                        k for k, v in self._backend._store.items()
                        if isinstance(v, dict) and
                        v.get('registered_at', 0) < cutoff]
                    for k in expired:
                        del self._backend._store[k]
                        removed += 1
            if removed > 0:
                logger.info(
                    f"GKD TTL: removed {removed} expired entries")
        except Exception as e:
            logger.debug(f"TTL enforce failed: {e}")

    # ── Primary API ────────────────────────────────────────────────────

    def lookup(
        self,
        token_ids: List[int],
        sequence_length: int,
    ) -> Optional[GKDHit]:
        """
        Look up whether a KV block for this token sequence already exists.

        Returns GKDHit on cache hit, None on cache miss.
        """
        with self._lock:
            self._total_lookups += 1

        content_hash = compute_hash(token_ids, sequence_length)
        raw = self._backend.get(content_hash)

        if raw is None:
            # LCP fallback — pipelined prefix lookup
            try:
                lcp_hit = self._pipelined_lcp_lookup(
                    token_ids, sequence_length)
                if lcp_hit is not None:
                    return lcp_hit
            except Exception as exc:
                logger.debug("GKD LCP lookup failed: %s", exc)

            with self._lock:
                self._cache_misses += 1
            return None

        # Collision verification
        with self._lock:
            self._collision_checks += 1

        stored_fingerprint = raw.get("fingerprint", [])
        if not _verify_fingerprint_fast(stored_fingerprint, token_ids):
            with self._lock:
                self._collision_detections += 1
            logger.error(
                "GKD COLLISION DETECTED for hash %s... "
                "Treating as miss.",
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

    def _pipelined_lcp_lookup(
        self,
        token_ids: List[int],
        sequence_length: int,
    ) -> Optional[GKDHit]:
        """
        LCP prefix lookup using pipelined backend reads.

        Instead of O(seq_len/128) sequential round-trips to Redis,
        this builds all prefix keys at once and fetches them in a
        single pipeline call. On a 1000-token sequence this reduces
        from ~8 sequential Redis calls to 1 pipelined call.
        """
        from memopt.cluster.prefix_index import (
            prefix_key, BLOCK_SIZE,
        )
        from memopt.cluster.prefix_index import \
            _verify_fingerprint_fast as vfp

        max_prefix = (sequence_length // BLOCK_SIZE) * BLOCK_SIZE
        if max_prefix == sequence_length:
            max_prefix -= BLOCK_SIZE
        if max_prefix < BLOCK_SIZE:
            return None

        # Build all prefix keys (longest first)
        lengths = list(range(max_prefix, BLOCK_SIZE - 1, -BLOCK_SIZE))
        keys = [prefix_key(token_ids, l) for l in lengths]

        # Pipelined fetch — one round-trip for all keys
        results = self._backend.pipeline_get(keys)

        # Find longest match with fingerprint verification
        for length, raw in zip(lengths, results):
            if raw is None:
                continue
            try:
                entry = json.loads(raw) if isinstance(raw, str) else raw
                stored_fp = entry.get("fingerprint", [])
                if not vfp(stored_fp, token_ids[:length]):
                    logger.debug(
                        "prefix_index: fingerprint mismatch len=%d",
                        length)
                    continue

                with self._lock:
                    self._cache_hits        += 1
                    self._lcp_hits          += 1
                    self._lcp_tokens_reused += length
                    self._lcp_tokens_total  += sequence_length

                return GKDHit(
                    block_ref   = entry["block_ref"],
                    node_id     = entry["node_id"],
                    hit_count   = 1,
                    matched_len = entry["matched_len"],
                    delta_start = entry["matched_len"],
                    is_partial  = True,
                )
            except Exception as exc:
                logger.debug(
                    "prefix_index: parse error len=%d: %s", length, exc)
                continue

        return None

    def register(
        self,
        token_ids: List[int],
        sequence_length: int,
        block_ref: str,
        node_id: str,
        size_bytes: Optional[int] = None,
        ttl_seconds: Optional[int] = None,
    ):
        """Register a newly computed KV block in the dedup store."""
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

        self._backend.set(
            content_hash, entry,
            ttl_seconds=ttl_seconds or self._ttl)

        try:
            from memopt.cluster.prefix_index import register_prefixes
            register_prefixes(
                token_ids, sequence_length, block_ref, node_id,
                self._backend)
        except Exception as exc:
            logger.debug("GKD prefix registration failed: %s", exc)

    # ── Output cache (for exact-hit compute skip) ───────────────────

    def get_output(self, block_ref: str) -> Optional[dict]:
        """Return cached output for a block_ref.
        Returns None if not found. Never raises."""
        try:
            with self._output_cache_lock:
                return self._output_cache.get(block_ref)
        except Exception:
            return None

    def register_output(self, block_ref: str, output: dict) -> None:
        """Store completion output alongside block_ref for future exact hits.
        Max 10K entries — evicts oldest on overflow. Never raises."""
        try:
            with self._output_cache_lock:
                if len(self._output_cache) >= 10_000:
                    oldest = next(iter(self._output_cache))
                    del self._output_cache[oldest]
                self._output_cache[block_ref] = output
        except Exception:
            pass

    def invalidate(self, token_ids: List[int], sequence_length: int):
        """Remove an entry from the store."""
        content_hash = compute_hash(token_ids, sequence_length)
        self._backend.delete(content_hash)

    # ── Instrumentation ────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
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
            "backend_degraded":            getattr(
                self._backend, "degraded", False),
            "backend_degraded_since":      getattr(
                self._backend, "degraded_since", None),
            "exact_hits":                  self._exact_hits,
            "lcp_hits":                    self._lcp_hits,
            "lcp_token_reuse_pct":         round(
                self._lcp_tokens_reused / self._lcp_tokens_total * 100, 1
            ) if self._lcp_tokens_total > 0 else 0.0,
            "total_hits":                  hits,
        }

    def reset_stats(self):
        with self._lock:
            self._total_lookups        = 0
            self._cache_hits           = 0
            self._cache_misses         = 0
            self._bytes_saved          = 0
            self._collision_checks     = 0
            self._collision_detections = 0
