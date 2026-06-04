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
import urllib.request
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from .bloom_filter import BloomFilter
from .hashing import compute_hash, make_fingerprint
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


class TenantGKDStats:
    """
    Per-tenant GKD statistics.

    Tracks hits, misses, tokens saved per tenant independently.
    Used by cost savings dashboard. Thread-safe.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stats: Dict[str, dict] = {}

    def record_hit(
        self, tenant_id: str, tokens_saved: int, is_partial: bool,
    ) -> None:
        with self._lock:
            s = self._get_or_create(tenant_id)
            if is_partial:
                s["partial_hits"] += 1
                s["partial_tokens_saved"] += tokens_saved
            else:
                s["exact_hits"] += 1
                s["exact_tokens_saved"] += tokens_saved

    def record_miss(self, tenant_id: str) -> None:
        with self._lock:
            s = self._get_or_create(tenant_id)
            s["misses"] += 1

    def get_tenant_stats(self, tenant_id: str) -> dict:
        with self._lock:
            s = self._get_or_create(tenant_id)
            total = s["exact_hits"] + s["partial_hits"] + s["misses"]
            hit_rate = (
                (s["exact_hits"] + s["partial_hits"]) / total * 100
                if total > 0 else 0.0)
            return {
                **s,
                "total_requests": total,
                "hit_rate_pct": round(hit_rate, 2),
            }

    def get_all_tenants(self) -> Dict[str, dict]:
        with self._lock:
            tenants = list(self._stats.keys())
        return {t: self.get_tenant_stats(t) for t in tenants}

    def _get_or_create(self, tenant_id: str) -> dict:
        """Must be called under self._lock."""
        if tenant_id not in self._stats:
            self._stats[tenant_id] = {
                "exact_hits": 0,
                "partial_hits": 0,
                "misses": 0,
                "exact_tokens_saved": 0,
                "partial_tokens_saved": 0,
            }
        return self._stats[tenant_id]


class HitRateWindow:
    """
    Rolling window hit rate counter.

    Tracks hit rate over the last N requests using a circular buffer.
    Gives a real-time hit rate that reflects current traffic — not a
    cumulative average that never resets.

    The 90% hit rate in the architecture doc came from a synthetic
    1000-user simulation. This class measures the ACTUAL hit rate
    on whatever traffic you are serving.
    """

    DEFAULT_WINDOW = 10_000

    def __init__(self, window_size: int = DEFAULT_WINDOW):
        self._window = window_size
        self._buf = bytearray(window_size)
        self._pos = 0
        self._total = 0
        self._hits = 0
        self._lock = threading.Lock()

    def record(self, is_hit: bool) -> None:
        with self._lock:
            idx = self._pos % self._window
            old = self._buf[idx]
            if self._total >= self._window and old == 1:
                self._hits -= 1

            val = 1 if is_hit else 0
            self._buf[idx] = val
            if val == 1:
                self._hits += 1

            self._pos += 1
            self._total += 1

    def hit_rate_pct(self) -> float:
        with self._lock:
            if self._total == 0:
                return 0.0
            n = min(self._total, self._window)
            return round(self._hits / n * 100, 2)

    def stats(self) -> dict:
        with self._lock:
            n = min(self._total, self._window)
            return {
                "window_size": self._window,
                "requests_seen": self._total,
                "window_filled": self._total >= self._window,
                "hit_rate_pct": round(
                    self._hits / n * 100, 2) if n > 0 else 0.0,
                "hits_in_window": self._hits,
                "note": (
                    "Hit rate measured on real traffic. "
                    "Synthetic benchmark showed 90%+. "
                    "Production rate depends on workload "
                    "prefix repetition."),
            }


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

        # Pod controller URL for two-tier lookup
        self._pod_url = os.environ.get(
            "MEMOPT_POD_CONTROLLER_URL", "")
        self._pod_hits = 0

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

        # Bloom filter: pre-filters backend queries.
        # If content_hash not in bloom → definitely not in store, skip backend.
        # Expected items configurable via env var (default 1M, 1% FP).
        bloom_expected = int(os.environ.get(
            "MEMOPT_BLOOM_EXPECTED_ITEMS", "1000000"))
        bloom_fp = float(os.environ.get(
            "MEMOPT_BLOOM_FP_RATE", "0.01"))
        self._bloom = BloomFilter(
            expected_items=bloom_expected,
            false_positive_rate=bloom_fp)
        self._bloom_filtered: int = 0

        # Output cache: block_ref → completion output dict.
        # Used by exact-hit path in serving/server.py to skip inference.
        # Max 10K entries; evicts oldest on overflow.
        self._output_cache: Dict[str, dict] = {}
        self._output_cache_lock = threading.Lock()

        # Per-tenant stats tracking
        self._tenant_stats = TenantGKDStats()

        # Rolling hit rate window
        self._hit_rate_window = HitRateWindow(
            window_size=int(os.environ.get(
                "MEMOPT_GKD_WINDOW_SIZE", "10000")))

        # Tenant isolation: when enabled, same tokens from different
        # tenants produce different hashes (complete data isolation)
        self._tenant_isolation = os.getenv(
            "MEMOPT_GKD_TENANT_ISOLATION", "false").lower() == "true"
        # Thread-local for passing tenant_id to internal LCP lookup
        self._current_lookup_tenant: str = ""

        if self._tenant_isolation:
            logger.info(
                "GKD: tenant isolation ENABLED "
                "(cross-tenant reuse disabled)")
        else:
            logger.info(
                "GKD: tenant isolation DISABLED "
                "(cross-tenant reuse enabled)")

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

    # ── Tenant-aware hashing ─────────────────────────────────────────

    def _make_hash(
        self,
        token_ids: List[int],
        seq_len: int,
        tenant_id: str = "",
    ) -> str:
        """
        Compute content hash.

        With tenant isolation: hash includes tenant_id. Same tokens from
        different tenants produce different hashes. Complete data isolation.

        Without tenant isolation: hash is token-only. Same tokens from any
        tenant produce the same hash. Maximum deduplication.
        """
        if (self._tenant_isolation
                and tenant_id
                and tenant_id != "_default"):
            import hashlib
            content = json.dumps(
                {"t": tenant_id, "k": token_ids[:seq_len]},
                separators=(",", ":"))
            return hashlib.sha256(content.encode()).hexdigest()
        return compute_hash(token_ids, seq_len)

    # ── Per-tenant stats ──────────────────────────────────────────────

    def tenant_stats(self, tenant_id: str = "") -> dict:
        """Return per-tenant stats. Empty tenant_id returns all tenants."""
        if tenant_id:
            return self._tenant_stats.get_tenant_stats(tenant_id)
        return self._tenant_stats.get_all_tenants()

    # ── Primary API ────────────────────────────────────────────────────

    def lookup(
        self,
        token_ids: List[int],
        sequence_length: int,
        tenant_id: str = "",
    ) -> Optional[GKDHit]:
        """
        Look up whether a KV block for this token sequence already exists.

        Returns GKDHit on cache hit, None on cache miss.
        """
        with self._lock:
            self._total_lookups += 1

        self._current_lookup_tenant = tenant_id or "_default"
        content_hash = self._make_hash(token_ids, sequence_length, tenant_id)

        # Bloom filter pre-check: if definitely absent, skip backend.get().
        # The bloom filter only tracks exact content hashes, not prefix
        # hashes, so the LCP fallback path must still run on bloom miss.
        if content_hash not in self._bloom:
            with self._lock:
                self._bloom_filtered += 1
            raw = None
        else:
            raw = self._backend.get(content_hash)

        if raw is None:
            # Tier 2: Try pod GKD cache before cluster store
            pod_hit = self._pod_lookup(
                content_hash, sequence_length)
            if pod_hit is not None:
                with self._lock:
                    self._cache_hits += 1
                    self._pod_hits += 1
                return pod_hit

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
            self._hit_rate_window.record(False)
            self._tenant_stats.record_miss(tenant_id or "_default")
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
        self._hit_rate_window.record(True)
        self._tenant_stats.record_hit(
            tenant_id or "_default",
            tokens_saved=sequence_length,
            is_partial=False)

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

        # Build all prefix keys (longest first). When tenant isolation is
        # enabled, namespace prefix keys by tenant — without this, an LCP
        # hit can match another tenant's registered prefix.
        lookup_tid = self._current_lookup_tenant
        prefix_tid = (lookup_tid
                      if (self._tenant_isolation
                          and lookup_tid
                          and lookup_tid != "_default")
                      else "")
        lengths = list(range(max_prefix, BLOCK_SIZE - 1, -BLOCK_SIZE))
        keys = [prefix_key(token_ids, l, prefix_tid) for l in lengths]

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

                self._hit_rate_window.record(True)
                self._tenant_stats.record_hit(
                    self._current_lookup_tenant or "_default",
                    tokens_saved=length,
                    is_partial=True)

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
        tenant_id: str = "",
    ):
        """Register a newly computed KV block in the dedup store."""
        content_hash = self._make_hash(token_ids, sequence_length, tenant_id)
        fingerprint  = make_fingerprint(token_ids)

        # Add to bloom filter so future lookups don't skip backend
        self._bloom.add(content_hash)

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
            prefix_tid = (tenant_id
                          if (self._tenant_isolation
                              and tenant_id
                              and tenant_id != "_default")
                          else "")
            register_prefixes(
                token_ids, sequence_length, block_ref, node_id,
                self._backend, tenant_id=prefix_tid)
        except Exception as exc:
            logger.debug("GKD prefix registration failed: %s", exc)

        # Register in pod cache (fire-and-forget)
        self._pod_register(
            content_hash, sequence_length,
            block_ref, node_id)

    # ── Agentic / workflow-scoped API ────────────────────────────────
    # The token-based lookup/register above is sized for chatbot traffic
    # (short TTL, no session). Long-running agents need:
    #   - keying by an opaque content_hash (caller already has bytes)
    #   - per-entry TTL (hours/days, not the global default)
    #   - workflow_id scoping so step 17 of an agent rehydrates step 1's KV
    # These methods write directly to the backend under a tenant-namespaced
    # key, bypassing prefix-index/LCP (an agent run wants exact match, not
    # partial prefix). Same backend, separate keyspace from register().

    def _namespace_hash(self, content_hash: str, tenant_id: str) -> str:
        """Mix tenant_id into the key so cross-tenant lookups miss."""
        if not tenant_id:
            return f"wf:v1:{content_hash}"
        import hashlib
        return "wf:v1:" + hashlib.sha256(
            f"{tenant_id}::{content_hash}".encode()).hexdigest()

    def register_workflow_block(
        self,
        content_hash: str,
        block_ref: str,
        node_id: str,
        size_bytes: int,
        tenant_id: str,
        workflow_id: str,
        ttl_seconds: int = 3600,
    ) -> bool:
        """
        Register a KV block scoped to a workflow.
        TTL defaults to 1 hour. Use 86400 for day-long agents.
        """
        now = time.time()
        key = self._namespace_hash(content_hash, tenant_id)
        entry = {
            "content_hash":  content_hash,
            "block_ref":     block_ref,
            "node_id":       node_id,
            "size_bytes":    size_bytes,
            "tenant_id":     tenant_id,
            "workflow_id":   workflow_id,
            "ttl_seconds":   ttl_seconds,
            "registered_at": now,
            "expires_at":    now + ttl_seconds if ttl_seconds > 0 else 0.0,
            "hit_count":     0,
        }
        self._backend.set(key, entry, ttl_seconds=ttl_seconds)
        return True

    def lookup_workflow(
        self,
        content_hash: str,
        tenant_id: str,
        workflow_id: str,
    ) -> Optional[GKDHit]:
        """
        Workflow-scoped lookup. Returns GKDHit on hit, None on miss.
        Miss conditions: not found, expired, wrong tenant, wrong workflow.
        Empty workflow_id on the entry means "shared across workflows".
        """
        key = self._namespace_hash(content_hash, tenant_id)
        raw = self._backend.get(key)
        if raw is None or not isinstance(raw, dict):
            return None

        # TTL check — evict on read to keep the local backend bounded
        # without waiting for the 5-minute sweep.
        expires_at = raw.get("expires_at", 0.0)
        if expires_at and time.time() > expires_at:
            self._backend.delete(key)
            return None

        stored_wf = raw.get("workflow_id", "")
        if stored_wf and workflow_id and stored_wf != workflow_id:
            return None

        self._backend.increment_hit(key)
        return GKDHit(
            block_ref  = raw["block_ref"],
            node_id    = raw["node_id"],
            size_bytes = raw.get("size_bytes", self._block_size_bytes),
            hit_count  = raw.get("hit_count", 0) + 1,
        )

    def evict_expired(self) -> int:
        """
        Sweep entries past expires_at. Returns count evicted.
        No-op for Redis (handled by setex). LocalGKDBackend only.
        """
        if not hasattr(self._backend, "_store"):
            return 0
        now = time.time()
        evicted = 0
        with self._backend._lock:
            for k in list(self._backend._store.keys()):
                v = self._backend._store.get(k)
                if not isinstance(v, dict):
                    continue
                exp = v.get("expires_at", 0.0)
                if exp and now > exp:
                    del self._backend._store[k]
                    evicted += 1
        return evicted

    def workflow_stats(self, workflow_id: str) -> dict:
        """Stats for blocks tagged with a given workflow_id."""
        if not hasattr(self._backend, "_store"):
            return {
                "workflow_id":     workflow_id,
                "total_blocks":    0,
                "active_blocks":   0,
                "bytes_cached_mb": 0.0,
                "expired_blocks":  0,
                "note": "workflow_stats requires LocalGKDBackend",
            }
        now = time.time()
        total = active = bytes_cached = 0
        with self._backend._lock:
            for v in self._backend._store.values():
                if not isinstance(v, dict):
                    continue
                if v.get("workflow_id", "") != workflow_id:
                    continue
                total += 1
                bytes_cached += v.get("size_bytes", 0)
                exp = v.get("expires_at", 0.0)
                if not exp or now < exp:
                    active += 1
        return {
            "workflow_id":     workflow_id,
            "total_blocks":    total,
            "active_blocks":   active,
            "bytes_cached_mb": round(bytes_cached / 1e6, 2),
            "expired_blocks":  total - active,
        }

    # ── Pod two-tier lookup ───────────────────────────────────────────

    def _pod_lookup(self, content_hash: str,
                    seq_len: int) -> Optional[GKDHit]:
        """Pod GKD cache lookup via HTTP. 10ms timeout. Never raises."""
        if not self._pod_url:
            return None
        try:
            url = (f"{self._pod_url}/pod/gkd/lookup"
                   f"?hash={content_hash}&seq_len={seq_len}")
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=0.050) as resp:
                data = json.loads(resp.read())
                if data.get("hit"):
                    return GKDHit(
                        block_ref=data.get("block_ref", ""),
                        node_id=data.get("node_id", ""),
                        is_partial=False)
        except Exception:
            pass
        return None

    def _pod_register(self, content_hash: str,
                      seq_len: int, block_ref: str,
                      node_id: str) -> None:
        """Register in pod cache. Fire-and-forget. Never blocks."""
        if not self._pod_url:
            return
        try:
            url = f"{self._pod_url}/pod/gkd/register"
            data = json.dumps({
                "hash": content_hash,
                "seq_len": seq_len,
                "block_ref": block_ref,
                "node_id": node_id,
            }).encode()
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST")
            threading.Thread(
                target=urllib.request.urlopen,
                args=(req,),
                kwargs={"timeout": 0.050},
                daemon=True).start()
        except Exception:
            pass

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
            "pod_hits":                    self._pod_hits,
            "pod_controller_url":          self._pod_url or "not_configured",
            "bloom_filtered":              self._bloom_filtered,
            "bloom_stats":                 self._bloom.stats(),
            "tenant_isolation":            self._tenant_isolation,
            "rolling_hit_rate":            self._hit_rate_window.stats(),
        }

    def reset_stats(self):
        with self._lock:
            self._total_lookups        = 0
            self._cache_hits           = 0
            self._cache_misses         = 0
            self._bytes_saved          = 0
            self._collision_checks     = 0
            self._collision_detections = 0


# ═══════════════════════════════════════════════════════════════════════════
# Backend factory — selects ScyllaDB → Redis → Local
# ═══════════════════════════════════════════════════════════════════════════

def make_gkd_backend(
    redis_url: str = "",
    node_id: str = "",
):
    """
    Create the best available GKD backend.

    Priority:
      1. ScyllaDB (SCYLLA_HOSTS env var set)
      2. Redis    (redis_url argument or REDIS_URL env var)
      3. Local    (in-memory, single-node)

    Never raises — always returns a working backend.
    """
    # 1. Try ScyllaDB
    scylla_hosts = os.environ.get("SCYLLA_HOSTS", "")
    if scylla_hosts:
        hosts = [h.strip() for h in scylla_hosts.split(",")
                 if h.strip()]
        if hosts:
            try:
                from memopt.cluster.scylla_gkd_backend import \
                    ScyllaGKDBackend
                backend = ScyllaGKDBackend(
                    hosts=hosts,
                    port=int(os.environ.get("SCYLLA_PORT", "9042")),
                    keyspace=os.environ.get(
                        "SCYLLA_KEYSPACE", "memopt"),
                    consistency=os.environ.get(
                        "SCYLLA_CONSISTENCY", "ONE"),
                    ttl_s=int(os.environ.get(
                        "MEMOPT_GKD_ENTRY_TTL_S", "3600")),
                )
                logger.info("GKD backend: ScyllaDB (%d hosts)", len(hosts))
                return backend
            except ImportError:
                logger.warning(
                    "ScyllaDB requested but cassandra-driver "
                    "not installed. pip install cassandra-driver. "
                    "Falling back.")
            except Exception as e:
                logger.warning(
                    "ScyllaDB connection failed: %s. Falling back.", e)

    # 2. Try Redis
    url = redis_url or os.environ.get("REDIS_URL", "")
    if url:
        try:
            backend = RedisGKDBackend(redis_url=url)
            logger.info("GKD backend: Redis")
            return backend
        except Exception as e:
            logger.warning("Redis failed: %s. Falling back to local.", e)

    # 3. Local fallback
    logger.info("GKD backend: local (single-node)")
    return LocalGKDBackend()
