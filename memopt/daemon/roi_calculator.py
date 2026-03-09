"""
ROI calculator for memopt optimizations.
Converts speedup into dollar savings per hour.

Formula:
    GPU hours saved = (1 - 1/speedup) * num_gpus * hours
    Dollar saved    = GPU hours saved * cost_per_gpu_hour

Example:
    LLaMA-13B on 2 GPUs, 2.0x speedup, $2.50/GPU/hr
    = (1 - 1/2.0) * 2 * 1hr * $2.50
    = 0.5 * 2 * $2.50 = $2.50/hr saved
    = $60/day = $1,825/year — on ONE model on ONE node
"""
from dataclasses import dataclass
from typing import List


@dataclass
class ROIEstimate:
    speedup_min: float
    speedup_max: float
    num_gpus: int
    gpu_cost_per_hour: float
    dollar_saved_per_hour_min: float
    dollar_saved_per_hour_max: float
    dollar_saved_per_day_min: float
    dollar_saved_per_day_max: float
    dollar_saved_per_year_min: float
    dollar_saved_per_year_max: float


class ClusterROICalculator:
    """
    Calculates dollar savings from optimization speedup.

    Usage:
        calc = ROICalculator(gpu_cost_per_hour=2.50)
        saved = calc.calculate(gpu_ids=[0, 1], speedup_min=1.5, speedup_max=2.5)
        print(f"${saved:.2f}/hr saved")
    """

    def __init__(self, gpu_cost_per_hour: float = 2.50):
        self.gpu_cost_per_hour = gpu_cost_per_hour

    def calculate(
        self,
        gpu_ids: List[int],
        speedup_min: float,
        speedup_max: float,
    ) -> float:
        """
        Returns conservative (min) dollar savings per hour.
        Uses min speedup for honest reporting — never inflate.
        """
        estimate = self.full_estimate(gpu_ids, speedup_min, speedup_max)
        return estimate.dollar_saved_per_hour_min

    def full_estimate(
        self,
        gpu_ids: List[int],
        speedup_min: float,
        speedup_max: float,
    ) -> ROIEstimate:
        """Full ROI breakdown with min/max and per-day/year projections."""
        num_gpus = len(gpu_ids)

        def savings(speedup: float) -> float:
            if speedup <= 1.0:
                return 0.0
            # Fraction of GPU time freed per wall-clock hour
            # 2x faster → same work in 0.5hr → 0.5 GPU-hours saved per GPU
            fraction_saved = 1.0 - (1.0 / speedup)
            return fraction_saved * num_gpus * self.gpu_cost_per_hour

        per_hour_min = savings(speedup_min)
        per_hour_max = savings(speedup_max)

        return ROIEstimate(
            speedup_min=speedup_min,
            speedup_max=speedup_max,
            num_gpus=num_gpus,
            gpu_cost_per_hour=self.gpu_cost_per_hour,
            dollar_saved_per_hour_min=round(per_hour_min, 2),
            dollar_saved_per_hour_max=round(per_hour_max, 2),
            dollar_saved_per_day_min=round(per_hour_min * 24, 2),
            dollar_saved_per_day_max=round(per_hour_max * 24, 2),
            dollar_saved_per_year_min=round(per_hour_min * 24 * 365, 2),
            dollar_saved_per_year_max=round(per_hour_max * 24 * 365, 2),
        )

    def cluster_roi(
        self,
        num_nodes: int,
        gpus_per_node: int,
        avg_speedup: float,
    ) -> dict:
        """
        Estimate cluster-wide ROI.

        Example:
            400 nodes × 8 GPUs × 2.0× speedup × $2.50/GPU/hr
            = $1,000/hr = $8.76M/year saved
        """
        total_gpus = num_nodes * gpus_per_node
        fraction_saved = 1.0 - (1.0 / max(avg_speedup, 1.001))
        per_hour = fraction_saved * total_gpus * self.gpu_cost_per_hour

        return {
            "total_gpus": total_gpus,
            "avg_speedup": avg_speedup,
            "dollar_saved_per_hour": round(per_hour, 2),
            "dollar_saved_per_day": round(per_hour * 24, 2),
            "dollar_saved_per_year": round(per_hour * 24 * 365, 2),
            "gpu_cost_per_hour": self.gpu_cost_per_hour,
        }


ROICalculator = ClusterROICalculator  # backward-compat alias
