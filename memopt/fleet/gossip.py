"""
Cross-Node Gossip Protocol — fleet-wide optimization knowledge base.

When a node successfully optimizes a model it publishes a "recipe" to the
control plane.  Every other node checks the knowledge base BEFORE running
test-measure-commit.  If a verified recipe exists it is applied immediately.

Result: first node takes ~5 minutes (discovery).
         every other node takes ~30 seconds (lookup + apply).
"""

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger("memopt.gossip")


@dataclass
class OptimizationRecipe:
    """
    A verified optimization config that worked on a specific
    model + GPU family combination.
    Shared across the entire fleet via the control plane.
    """
    recipe_hash:        str         # SHA256[:16] of model+gpu+dtype
    model_family:       str         # mistral, llama, falcon, etc.
    gpu_family:         str         # h100, a100, a10, rtx4090, etc.
    dtype:              str         # float16, bfloat16
    backend:            str         # vllm, tensorrt_llm
    batch_size:         int
    max_model_len:      int
    flash_attention:    str         # flash_attention_2, flash_attention_3
    gpu_memory_util:    float       # 0.0 – 1.0
    tensor_parallel:    int         # number of GPUs
    extra_flags:        Dict        # any additional vllm/trtllm flags

    # Proof of performance
    measured_speedup:   float       # actual measured speedup
    measured_tps:       float       # actual tok/s achieved
    verified_by_node:   str         # node that discovered this
    verified_at:        float       # unix timestamp
    verification_count: int = 1     # how many nodes confirmed this works

    @classmethod
    def compute_hash(cls, model_family: str,
                     gpu_family: str, dtype: str) -> str:
        """Deterministic 16-char hash for a model+gpu+dtype combination."""
        key = f"{model_family}:{gpu_family}:{dtype}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]


@dataclass
class GossipKnowledgeBase:
    """
    Local cache of optimization recipes.
    Stored in SQLite so it survives daemon restarts.
    Can be pointed at either the control-plane gossip.db
    (server side) or a per-node cache (daemon side).
    """
    db_path: str = field(
        default_factory=lambda: str(Path.home() / ".memopt" / "gossip.db")
    )

    def __post_init__(self):
        self._init_db()

    def _init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS recipes (
                    recipe_hash         TEXT PRIMARY KEY,
                    model_family        TEXT NOT NULL,
                    gpu_family          TEXT NOT NULL,
                    dtype               TEXT NOT NULL,
                    backend             TEXT NOT NULL,
                    batch_size          INTEGER,
                    max_model_len       INTEGER,
                    flash_attention     TEXT,
                    gpu_memory_util     REAL,
                    tensor_parallel     INTEGER,
                    extra_flags         TEXT,
                    measured_speedup    REAL,
                    measured_tps        REAL,
                    verified_by_node    TEXT,
                    verified_at         REAL,
                    verification_count  INTEGER DEFAULT 1,
                    local_cache_time    REAL
                )
            """)
            conn.commit()

    def lookup(self, model_family: str,
               gpu_family: str, dtype: str) -> Optional[OptimizationRecipe]:
        """
        Look up a verified recipe for this combination.
        Returns None if not found or if the cached entry is older than 7 days.
        """
        recipe_hash = OptimizationRecipe.compute_hash(model_family, gpu_family, dtype)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM recipes WHERE recipe_hash = ?",
                (recipe_hash,)
            ).fetchone()

        if not row:
            return None

        # Expire local cache entries older than 7 days
        cache_time = row[16]
        if time.time() - cache_time > 7 * 86400:
            logger.info("Recipe %s expired — will re-fetch from control plane", recipe_hash)
            return None

        return self._row_to_recipe(row)

    def store(self, recipe: OptimizationRecipe):
        """Insert or replace a recipe (upsert)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO recipes VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                recipe.recipe_hash, recipe.model_family, recipe.gpu_family,
                recipe.dtype, recipe.backend, recipe.batch_size,
                recipe.max_model_len, recipe.flash_attention,
                recipe.gpu_memory_util, recipe.tensor_parallel,
                json.dumps(recipe.extra_flags), recipe.measured_speedup,
                recipe.measured_tps, recipe.verified_by_node,
                recipe.verified_at, recipe.verification_count,
                time.time(),
            ))
            conn.commit()
        logger.info(
            "Recipe stored: %s on %s = %.2fx (hash %s)",
            recipe.model_family, recipe.gpu_family,
            recipe.measured_speedup, recipe.recipe_hash,
        )

    def upsert_publish(self, recipe_dict: dict) -> int:
        """
        Insert a new recipe or increment verification_count if it already exists.
        Returns the final verification_count.
        """
        rhash = recipe_dict["recipe_hash"]
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT verification_count FROM recipes WHERE recipe_hash = ?",
                (rhash,)
            ).fetchone()

            if existing:
                new_count = existing[0] + 1
                conn.execute(
                    "UPDATE recipes SET verification_count = ?, verified_at = ? "
                    "WHERE recipe_hash = ?",
                    (new_count, time.time(), rhash),
                )
                logger.info(
                    "Recipe %s confirmed by %s. Verification count: %d",
                    rhash, recipe_dict.get("verified_by_node"), new_count,
                )
                conn.commit()
                return new_count
            else:
                conn.execute("""
                    INSERT INTO recipes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    rhash,
                    recipe_dict["model_family"], recipe_dict["gpu_family"],
                    recipe_dict["dtype"], recipe_dict["backend"],
                    recipe_dict["batch_size"], recipe_dict["max_model_len"],
                    recipe_dict["flash_attention"], recipe_dict["gpu_memory_util"],
                    recipe_dict["tensor_parallel"],
                    json.dumps(recipe_dict.get("extra_flags", {})),
                    recipe_dict["measured_speedup"], recipe_dict["measured_tps"],
                    recipe_dict["verified_by_node"], recipe_dict["verified_at"],
                    recipe_dict.get("verification_count", 1),
                    time.time(),
                ))
                logger.info(
                    "New recipe stored: %s on %s = %.2fx",
                    recipe_dict["model_family"], recipe_dict["gpu_family"],
                    recipe_dict["measured_speedup"],
                )
                conn.commit()
                return recipe_dict.get("verification_count", 1)

    def all_recipes(self) -> List[OptimizationRecipe]:
        """Return all stored recipes, newest first."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM recipes ORDER BY verified_at DESC"
            ).fetchall()
        return [self._row_to_recipe(r) for r in rows]

    def stats(self) -> dict:
        """Return aggregate statistics over all stored recipes."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM recipes").fetchone()[0]
            avg_speed = conn.execute(
                "SELECT AVG(measured_speedup) FROM recipes"
            ).fetchone()[0] or 0.0
            top = conn.execute("""
                SELECT model_family, gpu_family, measured_speedup, verification_count
                FROM recipes ORDER BY measured_speedup DESC LIMIT 10
            """).fetchall()
        return {
            "total_recipes": total,
            "avg_speedup":   round(avg_speed, 2),
            "top_recipes": [
                {"model": r[0], "gpu": r[1],
                 "speedup": r[2], "verified_by": r[3]}
                for r in top
            ],
        }

    @staticmethod
    def _row_to_recipe(row) -> OptimizationRecipe:
        return OptimizationRecipe(
            recipe_hash=row[0], model_family=row[1], gpu_family=row[2],
            dtype=row[3], backend=row[4], batch_size=row[5],
            max_model_len=row[6], flash_attention=row[7],
            gpu_memory_util=row[8], tensor_parallel=row[9],
            extra_flags=json.loads(row[10] or "{}"),
            measured_speedup=row[11], measured_tps=row[12],
            verified_by_node=row[13], verified_at=row[14],
            verification_count=row[15],
        )


class GossipClient:
    """
    Runs on every daemon node.
    - Checks the knowledge base BEFORE running test-measure-commit.
    - Publishes successful optimizations to the control plane after discovery.
    - Pre-warms the local cache on startup via sync_all_recipes().
    """

    def __init__(self,
                 node_name: str,
                 control_plane_url: str,
                 api_key: str):
        self.node_name          = node_name
        self.control_plane_url  = control_plane_url.rstrip("/")
        self.api_key            = api_key
        self.kb                 = GossipKnowledgeBase()
        self._headers           = {"X-Memopt-API-Key": api_key}

    def check_before_optimize(self,
                               model_family: str,
                               gpu_family: str,
                               dtype: str = "float16") -> Optional[OptimizationRecipe]:
        """
        Call BEFORE running test-measure-commit.
        If a verified recipe exists: return it (apply directly, skip discovery).
        Returns None if no recipe found — caller should run normal discovery.
        """
        # 1. Local SQLite cache (instant — no network)
        recipe = self.kb.lookup(model_family, gpu_family, dtype)
        if recipe:
            logger.info(
                "GOSSIP HIT (local): %s on %s → %.2fx. "
                "Skipping test-measure-commit.",
                model_family, gpu_family, recipe.measured_speedup,
            )
            return recipe

        # 2. Pull from control plane
        try:
            import requests as _req
            recipe_hash = OptimizationRecipe.compute_hash(model_family, gpu_family, dtype)
            resp = _req.get(
                f"{self.control_plane_url}/api/v1/gossip/recipes/{recipe_hash}",
                headers=self._headers,
                timeout=10,
            )
            if resp.status_code == 200:
                recipe = OptimizationRecipe(**resp.json())
                self.kb.store(recipe)  # cache locally
                logger.info(
                    "GOSSIP HIT (control plane): %s on %s "
                    "verified by %s (%d nodes confirmed). Applying directly.",
                    model_family, gpu_family,
                    recipe.verified_by_node, recipe.verification_count,
                )
                return recipe
        except Exception as exc:
            logger.warning("Gossip pull failed (will discover locally): %s", exc)

        logger.info(
            "GOSSIP MISS: %s on %s. Running test-measure-commit locally.",
            model_family, gpu_family,
        )
        return None

    def publish_success(self,
                        model_family: str,
                        gpu_family: str,
                        dtype: str,
                        backend: str,
                        batch_size: int,
                        flash_attention: str,
                        gpu_memory_util: float,
                        tensor_parallel: int,
                        measured_speedup: float,
                        measured_tps: float,
                        max_model_len: int = 4096,
                        extra_flags: Dict = None):
        """
        Call AFTER a successful optimization with verified results.
        Publishes the recipe to the fleet so other nodes benefit immediately.
        """
        recipe = OptimizationRecipe(
            recipe_hash=OptimizationRecipe.compute_hash(model_family, gpu_family, dtype),
            model_family=model_family,
            gpu_family=gpu_family,
            dtype=dtype,
            backend=backend,
            batch_size=batch_size,
            max_model_len=max_model_len,
            flash_attention=flash_attention,
            gpu_memory_util=gpu_memory_util,
            tensor_parallel=tensor_parallel,
            extra_flags=extra_flags or {},
            measured_speedup=measured_speedup,
            measured_tps=measured_tps,
            verified_by_node=self.node_name,
            verified_at=time.time(),
            verification_count=1,
        )

        # Store locally first (works even if control plane is unreachable)
        self.kb.store(recipe)

        # Publish to control plane
        try:
            import requests as _req
            resp = _req.post(
                f"{self.control_plane_url}/api/v1/gossip/recipes",
                headers=self._headers,
                json=asdict(recipe),
                timeout=10,
            )
            if resp.status_code in (200, 201):
                logger.info(
                    "GOSSIP PUBLISH: %s/%s %.2fx recipe shared with fleet.",
                    model_family, gpu_family, measured_speedup,
                )
            else:
                logger.warning("Gossip publish failed: %s %s", resp.status_code, resp.text)
        except Exception as exc:
            logger.warning("Gossip publish failed (recipe saved locally): %s", exc)

    def sync_all_recipes(self):
        """
        Pull all recipes from control plane into local cache.
        Call on daemon startup to pre-warm the knowledge base.
        """
        try:
            import requests as _req
            resp = _req.get(
                f"{self.control_plane_url}/api/v1/gossip/recipes",
                headers=self._headers,
                timeout=15,
            )
            if resp.status_code == 200:
                recipes = resp.json().get("recipes", [])
                for r in recipes:
                    self.kb.store(OptimizationRecipe(**r))
                logger.info(
                    "Gossip sync: pulled %d recipes from control plane", len(recipes)
                )
        except Exception as exc:
            logger.warning("Gossip sync failed (will use local cache): %s", exc)
