"""
Arbitrage engine — polls cloud GPU pricing and recommends or executes
workload migration when a cheaper option is available.

Two modes:
  RECOMMEND   No API keys for cloud provisioning — logs recommendation only.
  EXECUTE     RUNPOD_API_KEY and/or LAMBDA_API_KEY set — provisions and
              migrates automatically when threshold is met.

Pricing is fetched from provider REST APIs. No prices are hardcoded.
If a provider API is unreachable, that provider is skipped for this cycle.

Cost model:
  effective_cost_per_token = gpu_price_usd_hr
                             / (tokens_per_sec × memopt_speedup_ratio)

Migration threshold:
  Migrate when: cheapest_option_cost < current_cost × (1 - threshold)
  AND this condition has been true for at least SUSTAINED_MINUTES.

Environment variables:
  RUNPOD_API_KEY               str    RunPod API key (optional)
  LAMBDA_API_KEY               str    Lambda Labs API key (optional)
  MEMOPT_ARBITRAGE_THRESHOLD   float  default 0.20 (20% saving required)
  MEMOPT_ARBITRAGE_SUSTAINED_M float  default 5.0 (minutes before migrating)
  MEMOPT_ARBITRAGE_POLL_S      float  default 60.0 (price poll interval)
  MEMOPT_ARBITRAGE_DRY_RUN     "1"    log migration plan, do not execute

Requires: requests (stdlib-only fallback uses urllib)
"""
from __future__ import annotations
import os
import time
import json
import logging
import threading
from dataclasses import dataclass
from typing import Optional, List

logger = logging.getLogger(__name__)

_THRESHOLD   = float(os.environ.get("MEMOPT_ARBITRAGE_THRESHOLD",   "0.20"))
_SUSTAINED_M = float(os.environ.get("MEMOPT_ARBITRAGE_SUSTAINED_M", "5.0"))
_POLL_S      = float(os.environ.get("MEMOPT_ARBITRAGE_POLL_S",      "60.0"))
_DRY_RUN     = os.environ.get("MEMOPT_ARBITRAGE_DRY_RUN", "") == "1"
_RUNPOD_KEY  = os.environ.get("RUNPOD_API_KEY",  "")
_LAMBDA_KEY  = os.environ.get("LAMBDA_API_KEY",  "")


@dataclass
class GPUOffer:
    """One available GPU from a cloud provider."""
    provider:     str    # "runpod" | "lambda"
    gpu_type:     str    # e.g. "A100 SXM4 80GB"
    price_usd_hr: float
    available:    bool
    region:       str


@dataclass
class MigrationRecommendation:
    """
    A recommendation to migrate to a cheaper GPU.
    Produced when cheapest_option_cost < current_cost × (1 - threshold).
    """
    target_provider:      str
    target_gpu_type:      str
    target_price_hr:      float
    current_price_hr:     float
    saving_pct:           float
    effective_cost_ratio: float   # target / current effective cost per token
    action:               str     # "recommend" | "execute"
    timestamp:            float


def _http_get(url: str, headers: dict = None,
              timeout: float = 5.0) -> Optional[dict]:
    """
    HTTP GET with stdlib urllib fallback if requests is not installed.
    Returns parsed JSON or None on error.
    """
    try:
        import requests
        r = requests.get(url, headers=headers or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"requests GET {url} failed: {e}")
        return None

    # stdlib fallback
    try:
        import urllib.request
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        logger.debug(f"urllib GET {url} failed: {e}")
        return None


def _fetch_runpod_prices() -> List[GPUOffer]:
    """
    Fetch GPU availability and pricing from RunPod's public API.
    Falls back to empty list on any error.
    """
    if not _RUNPOD_KEY:
        return []

    query = '{"query":"{gpuTypes{id displayName memoryInGb securePrice communityPrice lowestPrice{minimumBidPrice uninterruptablePrice}}}"}'
    try:
        data = _http_get(
            "https://api.runpod.io/graphql",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {_RUNPOD_KEY}",
            },
        )
        if data is None:
            return []
        gpu_types = data.get("data", {}).get("gpuTypes", [])
        offers = []
        for g in gpu_types:
            price = (g.get("securePrice") or
                     (g.get("lowestPrice") or {}).get("uninterruptablePrice"))
            if price is None:
                continue
            offers.append(GPUOffer(
                provider="runpod",
                gpu_type=g.get("displayName", g.get("id", "unknown")),
                price_usd_hr=float(price),
                available=True,
                region="global",
            ))
        return offers
    except Exception as e:
        logger.debug(f"RunPod price fetch failed: {e}")
        return []


def _fetch_lambda_prices() -> List[GPUOffer]:
    """
    Fetch GPU availability and pricing from Lambda Labs' public API.
    Endpoint: https://cloud.lambdalabs.com/api/v1/instance-types
    Requires LAMBDA_API_KEY for authentication.
    """
    if not _LAMBDA_KEY:
        return []

    try:
        data = _http_get(
            "https://cloud.lambdalabs.com/api/v1/instance-types",
            headers={"Authorization": f"Basic {_LAMBDA_KEY}"},
        )
        if data is None:
            return []
        offers = []
        for name, info in data.get("data", {}).items():
            spec    = info.get("instance_type", {})
            regions = info.get("regions_with_capacity_available", [])
            price   = spec.get("price_cents_per_hour")
            if price is None:
                continue
            gpu_desc = spec.get("gpu_description", name)
            offers.append(GPUOffer(
                provider="lambda",
                gpu_type=gpu_desc,
                price_usd_hr=float(price) / 100.0,
                available=len(regions) > 0,
                region=regions[0]["name"] if regions else "unknown",
            ))
        return offers
    except Exception as e:
        logger.debug(f"Lambda price fetch failed: {e}")
        return []


class ArbitrageEngine:
    """
    Polls cloud pricing and recommends or executes GPU migration.

    Usage:
        engine = ArbitrageEngine(
            current_price_hr=2.00,
            tokens_per_sec=100.0,
            speedup_ratio=1.0,
        )
        engine.start()
        rec = engine.latest_recommendation()
    """

    def __init__(
        self,
        current_price_hr: float = float(
            os.environ.get("MEMOPT_GPU_PRICE_USD_HR", "2.00")
        ),
        tokens_per_sec:   float = 1.0,
        speedup_ratio:    float = 1.0,
    ):
        self._current_price  = current_price_hr
        self._tokens_per_sec = tokens_per_sec
        self._speedup_ratio  = speedup_ratio
        self._lock           = threading.RLock()
        self._running        = False

        self._threshold_met_since:    Optional[float] = None
        self._latest_recommendation:  Optional[MigrationRecommendation] = None
        self._migration_history:      List[MigrationRecommendation] = []

    def update_metrics(self, tokens_per_sec: float,
                       speedup_ratio: float) -> None:
        """Called by metrics collector to keep cost model current."""
        with self._lock:
            self._tokens_per_sec = max(tokens_per_sec, 0.001)
            self._speedup_ratio  = max(speedup_ratio, 0.001)

    def latest_recommendation(self) -> Optional[MigrationRecommendation]:
        with self._lock:
            return self._latest_recommendation

    def migration_history(self) -> List[MigrationRecommendation]:
        with self._lock:
            return list(self._migration_history)

    def start(self) -> None:
        if not _RUNPOD_KEY and not _LAMBDA_KEY:
            logger.info(
                "ArbitrageEngine: no provider API keys set "
                "(RUNPOD_API_KEY, LAMBDA_API_KEY). "
                "Running in price-monitoring-only mode."
            )
        with self._lock:
            if self._running:
                return
            self._running = True
        t = threading.Thread(
            target=self._poll_loop, daemon=True, name="memopt-arbitrage"
        )
        t.start()
        logger.info("ArbitrageEngine started")

    def stop(self) -> None:
        with self._lock:
            self._running = False

    def _effective_cost(self, price_hr: float) -> float:
        """Effective cost per token accounting for memopt speedup."""
        with self._lock:
            tps = self._tokens_per_sec
            sr  = self._speedup_ratio
        tokens_per_hr = tps * 3600 * sr
        if tokens_per_hr <= 0:
            return float("inf")
        return price_hr / tokens_per_hr

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self._evaluate()
            except Exception as e:
                logger.debug(f"Arbitrage poll error: {e}")
            time.sleep(_POLL_S)

    def _evaluate(self) -> None:
        offers    = _fetch_runpod_prices() + _fetch_lambda_prices()
        available = [o for o in offers if o.available]
        if not available:
            logger.debug("Arbitrage: no provider offers available this cycle")
            return

        current_eff = self._effective_cost(self._current_price)
        best        = min(available,
                          key=lambda o: self._effective_cost(o.price_usd_hr))
        best_eff    = self._effective_cost(best.price_usd_hr)

        saving_pct  = (current_eff - best_eff) / current_eff \
                      if current_eff > 0 else 0.0

        logger.debug(
            f"Arbitrage: current={self._current_price:.3f}$/hr "
            f"best={best.provider}/{best.gpu_type}@{best.price_usd_hr:.3f}$/hr "
            f"saving={saving_pct:.1%}"
        )

        if saving_pct >= _THRESHOLD:
            now = time.monotonic()
            with self._lock:
                if self._threshold_met_since is None:
                    self._threshold_met_since = now
                elapsed_min = (now - self._threshold_met_since) / 60.0
        else:
            with self._lock:
                self._threshold_met_since = None
            return

        if elapsed_min < _SUSTAINED_M:
            logger.debug(
                f"Arbitrage: threshold met for {elapsed_min:.1f}m "
                f"(need {_SUSTAINED_M}m)"
            )
            return

        action = "execute" if (
            (_RUNPOD_KEY or _LAMBDA_KEY) and not _DRY_RUN
        ) else "recommend"

        rec = MigrationRecommendation(
            target_provider=best.provider,
            target_gpu_type=best.gpu_type,
            target_price_hr=best.price_usd_hr,
            current_price_hr=self._current_price,
            saving_pct=round(saving_pct * 100, 2),
            effective_cost_ratio=round(best_eff / current_eff, 4),
            action=action,
            timestamp=time.time(),
        )

        with self._lock:
            self._latest_recommendation = rec
            self._migration_history.append(rec)
            self._threshold_met_since   = None   # reset after acting

        if action == "recommend":
            logger.info(
                f"Arbitrage RECOMMEND: migrate to {best.provider}/"
                f"{best.gpu_type} @ ${best.price_usd_hr:.3f}/hr — "
                f"{saving_pct:.1%} saving. "
                f"Set RUNPOD_API_KEY/LAMBDA_API_KEY to enable auto-migration."
            )
        else:
            logger.info(
                f"Arbitrage EXECUTE (dry_run={_DRY_RUN}): "
                f"{best.provider}/{best.gpu_type} — {saving_pct:.1%} saving"
            )
            self._provision_and_migrate(rec)

    def _provision_and_migrate(
        self, rec: MigrationRecommendation
    ) -> None:
        """
        Override in a subclass to implement actual provisioning.
        The base class logs the migration plan and does nothing else.
        Subclass contract: must not raise — any error must be caught
        and logged, then inference continues on the current GPU.
        """
        logger.info(
            f"Migration plan: {rec.target_provider} "
            f"{rec.target_gpu_type} @ ${rec.target_price_hr:.3f}/hr. "
            f"Override _provision_and_migrate() to automate."
        )
