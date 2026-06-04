"""
GPU FinOps Intelligence — Pillar 7.

Tracks GPU utilization, calculates waste cost,
and produces auditable FinOps reports.

The core insight: average GPU utilization in
production is 5%. memopt measures real utilization
through the VMM allocator and produces signed
dollar amounts showing waste vs savings.
"""
import os
import time
import json
import threading
from dataclasses import dataclass, asdict
from typing import Optional, List
from datetime import datetime, timezone


# Default GPU costs ($/hr) by type.
# Override via MEMOPT_GPU_COST_PER_HOUR env var.
GPU_COSTS_PER_HOUR = {
    "H100":      3.50,
    "H100 80GB": 3.50,
    "A100":      2.00,
    "A100 80GB": 2.00,
    "A10G":      1.00,
    "L40S":      1.50,
    "RTX 6000":  1.20,
    "default":   2.00,
}


def get_gpu_cost_per_hour(gpu_name: str = "") -> float:
    """Get $/hr for a GPU type."""
    env_cost = os.environ.get("MEMOPT_GPU_COST_PER_HOUR")
    if env_cost:
        try:
            return float(env_cost)
        except ValueError:
            pass

    for key, cost in GPU_COSTS_PER_HOUR.items():
        if key.lower() in gpu_name.lower():
            return cost
    return GPU_COSTS_PER_HOUR["default"]


@dataclass
class UtilizationSample:
    """Single GPU utilization measurement."""
    timestamp:        float
    gpu_util_pct:     float   # 0-100
    memory_util_pct:  float   # 0-100
    power_watts:      float
    memory_used_gb:   float
    memory_total_gb:  float


@dataclass
class FinOpsReport:
    """
    Auditable GPU cost and waste report.
    Contains signed dollar amounts.
    """
    tenant_id:             str
    gpu_name:              str
    period_start:          str
    period_end:            str
    duration_hours:        float

    # Utilization
    avg_gpu_util_pct:      float
    avg_memory_util_pct:   float
    samples_collected:     int

    # Cost
    gpu_cost_per_hour:     float
    total_cost_usd:        float
    utilized_cost_usd:     float
    wasted_cost_usd:       float
    waste_pct:             float

    # memopt savings
    memopt_enabled:        bool
    kv_cache_hit_rate_pct: float
    estimated_savings_usd: float

    # Audit
    generated_at:          str
    energy_source:         str
    signature:             str = ""


class GPUFinOpsTracker:
    """
    Tracks GPU utilization and computes waste cost in real time.

    Usage:
        tracker = GPUFinOpsTracker(
            tenant_id="customer-1",
            gpu_cost_per_hour=2.00,
        )
        tracker.start()
        # ... workload runs ...
        report = tracker.get_report()
        tracker.stop()
    """

    def __init__(
        self,
        tenant_id: str = "default",
        gpu_idx: int = 0,
        sample_interval_s: float = 15.0,
        gpu_cost_per_hour: Optional[float] = None,
    ):
        self.tenant_id         = tenant_id
        self.gpu_idx           = gpu_idx
        self.sample_interval_s = sample_interval_s
        self._samples: List[UtilizationSample] = []
        self._lock             = threading.Lock()
        self._running          = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._gpu_name         = "unknown"
        self._gpu_cost_per_hour = gpu_cost_per_hour
        self._kv_hit_rate: float = 0.0

        # Try to init NVML
        self._nvml_available = False
        self._nvml_handle    = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
            name = pynvml.nvmlDeviceGetName(self._nvml_handle)
            self._gpu_name = name.decode() if isinstance(name, bytes) else name
            self._nvml_available = True
        except Exception:
            pass

        # Resolve cost from GPU name if not explicitly passed
        if self._gpu_cost_per_hour is None:
            self._gpu_cost_per_hour = get_gpu_cost_per_hour(self._gpu_name)

    def _sample(self) -> Optional[UtilizationSample]:
        """Take one utilization measurement."""
        if not self._nvml_available:
            return None
        try:
            import pynvml
            util  = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            mem   = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            power = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0

            return UtilizationSample(
                timestamp=time.time(),
                gpu_util_pct=float(util.gpu),
                memory_util_pct=float(util.memory),
                power_watts=power,
                memory_used_gb=mem.used / 1e9,
                memory_total_gb=mem.total / 1e9,
            )
        except Exception:
            return None

    def _poll_loop(self):
        """Background sampling thread."""
        while self._running:
            sample = self._sample()
            if sample:
                with self._lock:
                    self._samples.append(sample)
            time.sleep(self.sample_interval_s)

    def start(self):
        """Start background utilization sampling."""
        self._running    = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop sampling."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.sample_interval_s + 1)

    def record_sample(
        self,
        gpu_util_pct: float,
        memory_util_pct: float = 0.0,
        power_watts: float = 0.0,
        memory_used_gb: float = 0.0,
        memory_total_gb: float = 0.0,
    ):
        """Manually record a sample. Use when NVML is unavailable or for testing."""
        with self._lock:
            self._samples.append(
                UtilizationSample(
                    timestamp=time.time(),
                    gpu_util_pct=gpu_util_pct,
                    memory_util_pct=memory_util_pct,
                    power_watts=power_watts,
                    memory_used_gb=memory_used_gb,
                    memory_total_gb=memory_total_gb,
                )
            )

    def set_kv_hit_rate(self, hit_rate_pct: float):
        """
        Set the GKD hit rate for savings calculation.
        Call with the result from GKDStore.hit_rate().
        """
        self._kv_hit_rate = hit_rate_pct

    def get_stats(self) -> dict:
        """Current utilization stats."""
        with self._lock:
            samples = list(self._samples)

        if not samples:
            return {
                "samples":             0,
                "avg_gpu_util_pct":    0.0,
                "avg_memory_util_pct": 0.0,
                "avg_power_watts":     0.0,
                "gpu_name":            self._gpu_name,
                "nvml_available":      self._nvml_available,
            }

        n = len(samples)
        avg_gpu = sum(s.gpu_util_pct    for s in samples) / n
        avg_mem = sum(s.memory_util_pct for s in samples) / n
        avg_pwr = sum(s.power_watts     for s in samples) / n

        return {
            "samples":             n,
            "avg_gpu_util_pct":    round(avg_gpu, 1),
            "avg_memory_util_pct": round(avg_mem, 1),
            "avg_power_watts":     round(avg_pwr, 1),
            "gpu_name":            self._gpu_name,
            "nvml_available":      self._nvml_available,
        }

    def get_report(self, sign: bool = True) -> FinOpsReport:
        """Generate a FinOps report with waste cost calculation."""
        with self._lock:
            samples = list(self._samples)

        now        = time.time()
        start      = self._start_time or now
        duration_s = now - start
        duration_h = duration_s / 3600.0

        if samples:
            avg_gpu = sum(s.gpu_util_pct    for s in samples) / len(samples)
            avg_mem = sum(s.memory_util_pct for s in samples) / len(samples)
        else:
            avg_gpu = 0.0
            avg_mem = 0.0

        total_cost    = duration_h * self._gpu_cost_per_hour
        utilized_cost = total_cost * (avg_gpu / 100.0)
        wasted_cost   = total_cost - utilized_cost
        waste_pct     = (wasted_cost / total_cost * 100.0) if total_cost > 0 else 0.0

        # KV dedup hit rate means X% of prefill compute was avoided.
        # Prefill is ~30% of total inference cost.
        prefill_fraction = 0.30
        savings = total_cost * prefill_fraction * (self._kv_hit_rate / 100.0)

        report = FinOpsReport(
            tenant_id=self.tenant_id,
            gpu_name=self._gpu_name,
            period_start=datetime.fromtimestamp(start, tz=timezone.utc).isoformat(),
            period_end=datetime.fromtimestamp(now,   tz=timezone.utc).isoformat(),
            duration_hours=round(duration_h, 4),
            avg_gpu_util_pct=round(avg_gpu, 1),
            avg_memory_util_pct=round(avg_mem, 1),
            samples_collected=len(samples),
            gpu_cost_per_hour=self._gpu_cost_per_hour,
            total_cost_usd=round(total_cost, 4),
            utilized_cost_usd=round(utilized_cost, 4),
            wasted_cost_usd=round(wasted_cost, 4),
            waste_pct=round(waste_pct, 1),
            memopt_enabled=True,
            kv_cache_hit_rate_pct=round(self._kv_hit_rate, 1),
            estimated_savings_usd=round(savings, 4),
            generated_at=datetime.now(tz=timezone.utc).isoformat(),
            energy_source="nvml_measured" if self._nvml_available else "estimated",
        )

        if sign:
            report.signature = self._sign(report)

        return report

    def _sign(self, report: FinOpsReport) -> str:
        """HMAC-SHA256 sign the report."""
        import hashlib
        import hmac as _hmac

        key = os.environ.get("MEMOPT_SIGNING_KEY", "").encode()
        if not key:
            return "unsigned"

        data = asdict(report)
        data.pop("signature", None)
        payload = json.dumps(data, sort_keys=True).encode()
        sig = _hmac.new(key, payload, hashlib.sha256).hexdigest()
        return f"hmac-sha256:{sig}"

    def verify_signature(self, report: FinOpsReport) -> bool:
        """
        Verify a signed FinOpsReport.
        Returns True only if the HMAC matches the reconstructed payload.
        Mirrors _sign exactly: same field exclusion, same canonical JSON,
        same key source.
        """
        import hashlib
        import hmac as _hmac

        key = os.environ.get("MEMOPT_SIGNING_KEY", "").encode()
        if not key:
            return report.signature == "unsigned"

        sig = report.signature
        if not sig.startswith("hmac-sha256:"):
            return False

        provided = sig.split(":", 1)[1]

        data = asdict(report)
        data.pop("signature", None)
        payload = json.dumps(data, sort_keys=True).encode()
        expected = _hmac.new(key, payload, hashlib.sha256).hexdigest()

        return _hmac.compare_digest(provided, expected)

    def to_dict(self) -> dict:
        """Report as dict."""
        return asdict(self.get_report())

    def to_json(self) -> str:
        """Report as JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def __del__(self):
        try:
            if self._nvml_available:
                import pynvml
                pynvml.nvmlShutdown()
        except Exception:
            pass


def estimate_annual_waste(
    avg_util_pct: float,
    gpu_count: int,
    gpu_cost_per_hour: float = 2.00,
) -> dict:
    """
    Quick estimate of annual GPU waste.

    Args:
        avg_util_pct:      Average utilization 0-100
        gpu_count:         Number of GPUs in cluster
        gpu_cost_per_hour: Cost per GPU per hour

    Returns dict with cost breakdown.
    """
    hours_per_year = 8760
    total_cost = gpu_count * gpu_cost_per_hour * hours_per_year
    utilized   = total_cost * (avg_util_pct / 100.0)
    wasted     = total_cost - utilized

    return {
        "gpu_count":             gpu_count,
        "avg_utilization_pct":   avg_util_pct,
        "gpu_cost_per_hour":     gpu_cost_per_hour,
        "annual_total_cost_usd": round(total_cost, 2),
        "annual_utilized_usd":   round(utilized, 2),
        "annual_wasted_usd":     round(wasted, 2),
        "waste_pct":             round(100 - avg_util_pct, 1),
        "note": (
            "Based on average GPU utilization. "
            "Real savings depend on workload patterns."
        ),
    }
