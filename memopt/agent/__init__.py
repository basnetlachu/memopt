"""
memopt autonomous optimization agent.

Usage:
    from memopt.agent import MemoptAgent

    agent = MemoptAgent(target_speedup=2.0, max_rounds=10)
    report = agent.run(model, sample_input)
    print(report.final_speedup)
    # use report.optimized_model for inference
"""

from .optimization_agent import MemoptAgent, AgentReport, AgentRound
from .training_agent import (
    TrainingAgent,
    TrainingAgentReport,
    apply_gradient_checkpointing,
    apply_bf16_mixed_precision,
    TRAINING_OPTIMIZATION_PRIORITY,
    TRAINING_BLACKLIST,
)

__all__ = [
    "MemoptAgent",
    "AgentReport",
    "AgentRound",
    "TrainingAgent",
    "TrainingAgentReport",
    "apply_gradient_checkpointing",
    "apply_bf16_mixed_precision",
    "TRAINING_OPTIMIZATION_PRIORITY",
    "TRAINING_BLACKLIST",
]
