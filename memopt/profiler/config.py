"""
Configuration management for memopt.

Provides centralized configuration with sensible defaults and override support.
Supports loading from YAML/JSON files or programmatic configuration.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("memopt")


@dataclass
class AttributionConfig:
    """Configuration for traffic attribution engine."""
    min_read_count: int = 2
    max_reuse_distance: int = 3
    cache_thrashing_threshold: float = 1.0
    temporal_locality_threshold: float = 0.3
    min_tensor_size_bytes: int = 1024  # Skip tiny tensors


@dataclass
class MeasurementConfig:
    """Configuration for performance measurement."""
    warmup_iterations: int = 5
    measure_iterations: int = 20
    min_improvement: float = 0.02  # 2% minimum to commit
    outlier_threshold: float = 2.0  # IQR multiplier


@dataclass
class VerificationConfig:
    """Configuration for semantic verification."""
    fp16_rtol: float = 1e-2
    fp16_atol: float = 1e-2
    fp32_rtol: float = 1e-3
    fp32_atol: float = 1e-3
    fp64_rtol: float = 1e-5
    fp64_atol: float = 1e-5
    sample_size: int = 5
    auto_tolerance: bool = True


@dataclass
class CompileConfig:
    """Configuration for torch.compile behavior."""
    enabled: bool = True
    default_mode: str = "default"
    fallback_mode: str = "reduce-overhead"
    max_retries: int = 2
    fullgraph: bool = False
    # Relaxed tolerance for compiled models (they can have small numerical differences)
    compiled_rtol_multiplier: float = 2.0
    compiled_atol_multiplier: float = 2.0


@dataclass
class MemoptConfig:
    """Complete memopt configuration."""
    attribution: AttributionConfig = field(default_factory=AttributionConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)

    # Global settings
    use_fallbacks: bool = True
    log_level: str = "INFO"
    knowledge_base_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoptConfig":
        """Create config from dictionary."""
        config = cls()

        if "attribution" in data:
            for k, v in data["attribution"].items():
                if hasattr(config.attribution, k):
                    setattr(config.attribution, k, v)

        if "measurement" in data:
            for k, v in data["measurement"].items():
                if hasattr(config.measurement, k):
                    setattr(config.measurement, k, v)

        if "verification" in data:
            for k, v in data["verification"].items():
                if hasattr(config.verification, k):
                    setattr(config.verification, k, v)

        if "compile" in data:
            for k, v in data["compile"].items():
                if hasattr(config.compile, k):
                    setattr(config.compile, k, v)

        for key in ["use_fallbacks", "log_level", "knowledge_base_path"]:
            if key in data:
                setattr(config, key, data[key])

        return config

    @classmethod
    def load(cls, path: Path) -> "MemoptConfig":
        """Load config from file (JSON or YAML)."""
        path = Path(path)

        if not path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return cls()

        try:
            with open(path) as f:
                if path.suffix in (".yaml", ".yml"):
                    try:
                        import yaml
                        data = yaml.safe_load(f)
                    except ImportError:
                        logger.warning("PyYAML not installed, cannot load YAML config")
                        return cls()
                else:
                    data = json.load(f)

            logger.info(f"Loaded config from {path}")
            return cls.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load config from {path}: {e}")
            return cls()

    def save(self, path: Path):
        """Save config to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            if path.suffix in (".yaml", ".yml"):
                try:
                    import yaml
                    yaml.dump(self.to_dict(), f, default_flow_style=False)
                except ImportError:
                    json.dump(self.to_dict(), f, indent=2)
            else:
                json.dump(self.to_dict(), f, indent=2)


# Global config instance
_global_config: Optional[MemoptConfig] = None


def get_config() -> MemoptConfig:
    """Get global config instance."""
    global _global_config
    if _global_config is None:
        _global_config = MemoptConfig()
    return _global_config


def set_config(config: MemoptConfig):
    """Set global config instance."""
    global _global_config
    _global_config = config


def load_config(path: Path) -> MemoptConfig:
    """Load and set global config from file."""
    config = MemoptConfig.load(path)
    set_config(config)
    return config
